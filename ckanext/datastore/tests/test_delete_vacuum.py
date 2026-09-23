"""Regression tests for delayed VACUUM after committed record deletions."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest
import sqlalchemy as sa
from freezegun import freeze_time
from redis.exceptions import ConnectionError as RedisConnectionError
from rq.job import Job
from rq.registry import ScheduledJobRegistry
from rq.scheduler import RQScheduler

from ckan import model
from ckan.lib import jobs
from ckan.plugins.toolkit import ObjectNotFound
from ckan.tests import factories, helpers
from ckanext.datastore.backend import postgres as db


pytestmark = [
    pytest.mark.ckan_config("ckan.plugins", "datastore"),
    pytest.mark.usefixtures(
        "clean_datastore", "with_plugins", "clean_redis", "with_test_worker"
    ),
]

DELETE_ACTIONS = ["datastore_delete", "datastore_records_delete"]


def create_table():
    resource = factories.Resource()
    helpers.call_action(
        "datastore_create",
        resource_id=resource["id"],
        force=True,
        fields=[{"id": "value", "type": "int"}],
        records=[{"value": 1}, {"value": 2}, {"value": 3}],
    )
    return resource["id"]


@pytest.fixture
def table():
    return create_table()


@pytest.fixture
def queue():
    return jobs.get_queue()


@pytest.fixture
def clock():
    with freeze_time(datetime.now(timezone.utc).replace(microsecond=0)) as frozen:
        yield frozen


@pytest.fixture
def vacuum_statements():
    """Observe completed SQL without replacing the database or VACUUM."""
    statements = []
    # CKAN workers dispose and recreate the cached DataStore engine.
    engine = sa.engine.Engine

    def record_vacuum(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("VACUUM"):
            statements.append(statement)

    sa.event.listen(engine, "after_cursor_execute", record_vacuum)
    try:
        yield statements
    finally:
        sa.event.remove(engine, "after_cursor_execute", record_vacuum)


def run_due_jobs(queue):
    scheduler = RQScheduler([queue], connection=queue.connection)
    scheduler.prepare_registries([queue.name])
    scheduler.enqueue_scheduled_jobs()
    jobs.Worker().work(burst=True)
    assert queue.failed_job_registry.count == 0


@pytest.mark.parametrize("action", DELETE_ACTIONS)
@pytest.mark.parametrize(
    "filters, remaining", [({"value": 1}, [2, 3]), ({}, [])],
    ids=["matching-filter", "all-records"],
)
def test_records_deleted_schedule_delayed_vacuum(
    action, filters, remaining, table, queue, clock, vacuum_statements
):
    helpers.call_action(action, resource_id=table, force=True, filters=filters)

    result = helpers.call_action("datastore_search", resource_id=table)
    assert [record["value"] for record in result["records"]] == remaining

    scheduled = queue.scheduled_job_registry
    job_ids = scheduled.get_job_ids()
    assert len(job_ids) == 1
    assert scheduled.get_scheduled_time(job_ids[0]) == (
        datetime.now(timezone.utc) + timedelta(seconds=10)
    )
    assert queue.count == 0
    assert vacuum_statements == []

    clock.tick(timedelta(seconds=9))
    run_due_jobs(queue)
    assert vacuum_statements == []

    clock.tick(timedelta(seconds=1))
    run_due_jobs(queue)
    assert len(vacuum_statements) == 1
    assert db.identifier(table) in vacuum_statements[0]
    assert scheduled.get_job_ids() == []


def test_redis_failure_does_not_fail_committed_delete(
    table, queue, monkeypatch
):
    def fail_transaction(*args, **kwargs):
        raise RedisConnectionError("Redis unavailable")

    monkeypatch.setattr(queue.connection, "transaction", fail_transaction)
    with helpers.recorded_logs(db.log) as logs:
        result = helpers.call_action(
            "datastore_delete", resource_id=table, force=True,
            filters={"value": 1},
        )
    assert result["resource_id"] == table

    remaining = helpers.call_action("datastore_search", resource_id=table)
    assert [record["value"] for record in remaining["records"]] == [2, 3]
    assert queue.scheduled_job_registry.get_job_ids() == []
    logs.assert_log(
        "error",
        f"Could not schedule VACUUM for resource {table} after deletion",
    )


def test_dropping_table_does_not_schedule_vacuum(
    table, queue, vacuum_statements
):
    helpers.call_action("datastore_delete", resource_id=table, force=True)

    with pytest.raises(ObjectNotFound):
        helpers.call_action("datastore_search", resource_id=table)

    assert queue.scheduled_job_registry.get_job_ids() == []
    assert queue.count == 0
    assert vacuum_statements == []


@pytest.mark.parametrize("action", DELETE_ACTIONS)
def test_dropping_table_with_pending_vacuum_is_harmless(
    action, table, queue, clock
):
    helpers.call_action(
        action, resource_id=table, force=True, filters={"value": 1}
    )
    assert len(queue.scheduled_job_registry.get_job_ids()) == 1

    helpers.call_action("datastore_delete", resource_id=table, force=True)

    clock.tick(timedelta(seconds=10))
    run_due_jobs(queue)
    assert queue.scheduled_job_registry.get_job_ids() == []
    assert queue.count == 0
    with pytest.raises(ObjectNotFound):
        helpers.call_action("datastore_search", resource_id=table)


@pytest.mark.parametrize("action", DELETE_ACTIONS)
def test_unmatched_filter_does_not_schedule_vacuum(
    action, table, queue, vacuum_statements
):
    helpers.call_action(
        action, resource_id=table, force=True, filters={"value": 99}
    )

    result = helpers.call_action("datastore_search", resource_id=table)
    assert [record["value"] for record in result["records"]] == [1, 2, 3]
    assert queue.scheduled_job_registry.get_job_ids() == []
    assert queue.count == 0
    assert vacuum_statements == []


@pytest.mark.parametrize("first_action", DELETE_ACTIONS)
@pytest.mark.parametrize("second_action", DELETE_ACTIONS)
def test_unmatched_filter_does_not_reset_pending_vacuum(
    first_action, second_action, table, queue, clock, vacuum_statements
):
    helpers.call_action(
        first_action, resource_id=table, force=True, filters={"value": 1}
    )
    scheduled = queue.scheduled_job_registry
    job_ids = scheduled.get_job_ids()
    assert len(job_ids) == 1
    deadline = datetime.now(timezone.utc) + timedelta(seconds=10)

    clock.tick(timedelta(seconds=5))
    helpers.call_action(
        second_action, resource_id=table, force=True, filters={"value": 99}
    )

    result = helpers.call_action("datastore_search", resource_id=table)
    assert [record["value"] for record in result["records"]] == [2, 3]
    assert scheduled.get_job_ids() == job_ids
    assert scheduled.get_scheduled_time(job_ids[0]) == deadline
    assert vacuum_statements == []

    clock.tick(timedelta(seconds=5))
    run_due_jobs(queue)
    assert len(vacuum_statements) == 1
    assert db.identifier(table) in vacuum_statements[0]
    assert scheduled.get_job_ids() == []


@pytest.mark.parametrize("first_action", DELETE_ACTIONS)
@pytest.mark.parametrize("second_action", DELETE_ACTIONS)
def test_deletions_within_cooldown_vacuum_once(
    first_action, second_action, table, queue, clock, vacuum_statements
):
    helpers.call_action(
        first_action, resource_id=table, force=True, filters={"value": 1}
    )
    clock.tick(timedelta(seconds=5))
    helpers.call_action(
        second_action, resource_id=table, force=True, filters={"value": 2}
    )

    result = helpers.call_action("datastore_search", resource_id=table)
    assert [record["value"] for record in result["records"]] == [3]
    scheduled = queue.scheduled_job_registry
    job_ids = scheduled.get_job_ids()
    assert len(job_ids) == 1
    assert scheduled.get_scheduled_time(job_ids[0]) == (
        datetime.now(timezone.utc) + timedelta(seconds=10)
    )

    # The first deadline has passed, but the second deletion reset it.
    clock.tick(timedelta(seconds=6))
    run_due_jobs(queue)
    assert vacuum_statements == []

    clock.tick(timedelta(seconds=4))
    run_due_jobs(queue)
    assert len(vacuum_statements) == 1
    assert db.identifier(table) in vacuum_statements[0]
    assert scheduled.get_job_ids() == []

    clock.tick(timedelta(seconds=10))
    run_due_jobs(queue)
    assert len(vacuum_statements) == 1


@pytest.mark.parametrize("first_action", DELETE_ACTIONS)
@pytest.mark.parametrize("second_action", DELETE_ACTIONS)
def test_deletion_after_completed_vacuum_schedules_another(
    first_action, second_action, table, queue, clock, vacuum_statements
):
    helpers.call_action(
        first_action, resource_id=table, force=True, filters={"value": 1}
    )
    clock.tick(timedelta(seconds=10))
    run_due_jobs(queue)
    assert len(vacuum_statements) == 1
    assert queue.scheduled_job_registry.get_job_ids() == []

    helpers.call_action(
        second_action, resource_id=table, force=True, filters={"value": 2}
    )
    scheduled = queue.scheduled_job_registry
    job_ids = scheduled.get_job_ids()
    assert len(job_ids) == 1
    assert scheduled.get_scheduled_time(job_ids[0]) == (
        datetime.now(timezone.utc) + timedelta(seconds=10)
    )

    clock.tick(timedelta(seconds=9))
    run_due_jobs(queue)
    assert len(vacuum_statements) == 1

    clock.tick(timedelta(seconds=1))
    run_due_jobs(queue)
    assert len(vacuum_statements) == 2
    assert all(db.identifier(table) in statement for statement in vacuum_statements)
    assert scheduled.get_job_ids() == []
    assert queue.count == 0
    result = helpers.call_action("datastore_search", resource_id=table)
    assert [record["value"] for record in result["records"]] == [3]


def test_different_resources_each_get_vacuum(
    table, queue, clock, vacuum_statements
):
    other_table = create_table()
    for resource_id in [table, other_table]:
        helpers.call_action(
            "datastore_delete",
            resource_id=resource_id,
            force=True,
            filters={"value": 1},
        )

    assert len(queue.scheduled_job_registry.get_job_ids()) == 2

    clock.tick(timedelta(seconds=10))
    run_due_jobs(queue)
    assert len(vacuum_statements) == 2
    for resource_id in [table, other_table]:
        assert sum(
            db.identifier(resource_id) in statement
            for statement in vacuum_statements
        ) == 1


@pytest.mark.parametrize("first_action", DELETE_ACTIONS)
@pytest.mark.parametrize("second_action", DELETE_ACTIONS)
@pytest.mark.parametrize("already_scheduled", [False, True])
def test_overlapping_deletions_vacuum_once(
    first_action, second_action, already_scheduled, table, queue, clock,
    vacuum_statements, test_request_context
):
    if already_scheduled:
        helpers.call_action(
            "datastore_delete", resource_id=table, force=True,
            filters={"value": 1},
        )
        clock.tick(timedelta(seconds=5))

    barrier = Barrier(2, timeout=10)
    engine = db.get_write_engine()

    def overlap_deletes(conn, cursor, statement, parameters, context,
                        executemany):
        if statement.startswith(f'DELETE FROM {db.identifier(table)}'):
            barrier.wait()

    def delete(action, value):
        with test_request_context():
            try:
                return helpers.call_action(
                    action, resource_id=table, force=True,
                    filters={"value": value},
                )
            finally:
                model.Session.remove()

    sa.event.listen(engine, "before_cursor_execute", overlap_deletes)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            values = [2, 3] if already_scheduled else [1, 2]
            futures = [
                executor.submit(delete, action, value)
                for action, value in zip([first_action, second_action], values)
            ]
            for future in futures:
                future.result(timeout=15)
    finally:
        sa.event.remove(engine, "before_cursor_execute", overlap_deletes)

    result = helpers.call_action("datastore_search", resource_id=table)
    assert [record["value"] for record in result["records"]] == (
        [] if already_scheduled else [3]
    )
    assert len(queue.scheduled_job_registry.get_job_ids()) == 1

    clock.tick(timedelta(seconds=9))
    run_due_jobs(queue)
    assert vacuum_statements == []

    clock.tick(timedelta(seconds=1))
    run_due_jobs(queue)
    assert len(vacuum_statements) == 1
    assert queue.scheduled_job_registry.get_job_ids() == []


@pytest.mark.parametrize("stage", ["selected", "fetched"])
def test_deletion_while_scheduler_moves_due_job_resets_cooldown(
    stage, table, queue, clock, vacuum_statements, monkeypatch
):
    helpers.call_action(
        "datastore_delete", resource_id=table, force=True, filters={"value": 1}
    )
    clock.tick(timedelta(seconds=10))
    interleaved = []

    def delete_more():
        helpers.call_action(
            "datastore_records_delete", resource_id=table, force=True,
            filters={"value": 2},
        )
        interleaved.append(True)

    # Pause the real scheduler after it has selected or fetched the old job.
    # A deletion then replaces that job before the scheduler enqueues it.
    with monkeypatch.context() as patch:
        if stage == "selected":
            original = ScheduledJobRegistry.get_jobs_to_schedule

            def select_then_delete(self, *args, **kwargs):
                job_ids = original(self, *args, **kwargs)
                delete_more()
                return job_ids

            patch.setattr(
                ScheduledJobRegistry, "get_jobs_to_schedule", select_then_delete
            )
        else:
            original = Job.fetch_many

            def fetch_then_delete(*args, **kwargs):
                fetched = original(*args, **kwargs)
                delete_more()
                return fetched

            patch.setattr(Job, "fetch_many", fetch_then_delete)

        run_due_jobs(queue)

    assert interleaved == [True]
    result = helpers.call_action("datastore_search", resource_id=table)
    assert [record["value"] for record in result["records"]] == [3]
    assert vacuum_statements == []
    assert len(queue.scheduled_job_registry.get_job_ids()) == 1

    clock.tick(timedelta(seconds=9))
    run_due_jobs(queue)
    assert vacuum_statements == []

    clock.tick(timedelta(seconds=1))
    run_due_jobs(queue)
    assert len(vacuum_statements) == 1
    assert queue.scheduled_job_registry.get_job_ids() == []

    clock.tick(timedelta(seconds=10))
    run_due_jobs(queue)
    assert len(vacuum_statements) == 1
