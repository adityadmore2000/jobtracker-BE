"""Tests for company normalization, get_or_create_company, and multi-company scenarios."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.company_matching import normalize_company_name
from app.company_resolution import get_company_by_normalized_name, get_or_create_company
from app.models import Company, JobApplication
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget


# ---------------------------------------------------------------------------
# normalize_company_name
# ---------------------------------------------------------------------------

def test_normalize_trims_whitespace():
    assert normalize_company_name("  Rockwell  ") == "rockwell"


def test_normalize_collapses_internal_spaces():
    assert normalize_company_name("Rockwell  Automation") == "rockwell automation"


def test_normalize_casefolds():
    assert normalize_company_name("ROCKWELL") == "rockwell"
    assert normalize_company_name("rockwell") == "rockwell"
    assert normalize_company_name("Rockwell") == "rockwell"


def test_normalize_all_case_variants_match():
    assert normalize_company_name(" Rockwell ") == normalize_company_name("rockwell")
    assert normalize_company_name("ROCKWELL") == normalize_company_name("rockwell")


def test_different_names_stay_separate():
    assert normalize_company_name("Rockwell") != normalize_company_name("Rockwell Automation")


# ---------------------------------------------------------------------------
# get_or_create_company
# ---------------------------------------------------------------------------

def test_get_or_create_creates_new(db_session: Session):
    company = get_or_create_company(db_session, "Rockwell")
    db_session.commit()
    assert company.id is not None
    assert company.name == "Rockwell"
    assert company.normalized_name == "rockwell"


def test_get_or_create_reuses_same_normalized_name(db_session: Session):
    c1 = get_or_create_company(db_session, "Rockwell")
    db_session.commit()
    c2 = get_or_create_company(db_session, "rockwell")
    db_session.commit()
    assert c1.id == c2.id


def test_get_or_create_reuses_case_variant(db_session: Session):
    c1 = get_or_create_company(db_session, "ROCKWELL")
    db_session.commit()
    c2 = get_or_create_company(db_session, " Rockwell ")
    db_session.commit()
    assert c1.id == c2.id


def test_get_or_create_separate_for_different_names(db_session: Session):
    c1 = get_or_create_company(db_session, "Rockwell")
    db_session.commit()
    c2 = get_or_create_company(db_session, "Rockwell Automation")
    db_session.commit()
    assert c1.id != c2.id


def test_get_or_create_blank_raises(db_session: Session):
    with pytest.raises(ValueError):
        get_or_create_company(db_session, "")


def test_get_or_create_whitespace_only_raises(db_session: Session):
    with pytest.raises(ValueError):
        get_or_create_company(db_session, "   ")


# ---------------------------------------------------------------------------
# Company reuse — same company, multiple roles
# ---------------------------------------------------------------------------

def test_same_company_different_roles_creates_two_applications_one_company(db_session: Session):
    p1 = MutationPayload(
        operation="create_draft",
        target=MutationTarget(),
        changes=ApplicationChanges(company="Rockwell", role="AI Engineer Intern"),
    )
    r1 = dispatch(p1, db_session)
    assert r1.success

    save1 = MutationPayload(
        operation="save_draft",
        target=MutationTarget(draft_id=str(r1.draft["id"])),
        changes=ApplicationChanges(),
    )
    dispatch(save1, db_session)

    p2 = MutationPayload(
        operation="create_draft",
        target=MutationTarget(),
        changes=ApplicationChanges(company="Rockwell", role="Graduate Engineer Trainee"),
    )
    r2 = dispatch(p2, db_session)
    assert r2.success

    save2 = MutationPayload(
        operation="save_draft",
        target=MutationTarget(draft_id=str(r2.draft["id"])),
        changes=ApplicationChanges(),
    )
    dispatch(save2, db_session)

    apps = db_session.query(JobApplication).filter(JobApplication.is_draft == False).all()  # noqa: E712
    assert len(apps) == 2
    companies = db_session.query(Company).all()
    assert len(companies) == 1
    assert companies[0].name == "Rockwell"


def test_same_company_different_case_creates_one_company(db_session: Session):
    p1 = MutationPayload(
        operation="create_draft",
        target=MutationTarget(),
        changes=ApplicationChanges(company="Neilsoft", role="AI Engineer"),
    )
    r1 = dispatch(p1, db_session)
    dispatch(MutationPayload(operation="save_draft", target=MutationTarget(draft_id=str(r1.draft["id"])), changes=ApplicationChanges()), db_session)

    p2 = MutationPayload(
        operation="create_draft",
        target=MutationTarget(),
        changes=ApplicationChanges(company="neilsoft", role="ML Engineer"),
    )
    r2 = dispatch(p2, db_session)
    dispatch(MutationPayload(operation="save_draft", target=MutationTarget(draft_id=str(r2.draft["id"])), changes=ApplicationChanges()), db_session)

    companies = db_session.query(Company).all()
    assert len(companies) == 1


# ---------------------------------------------------------------------------
# Public DTO contract: company and role exposed, company_id absent
# ---------------------------------------------------------------------------

def test_application_dict_exposes_company_and_role_not_company_id(db_session: Session):
    from app.mutation_dispatcher import _application_to_dict

    p = MutationPayload(
        operation="create_draft",
        target=MutationTarget(),
        changes=ApplicationChanges(company="Acme", role="LLM Inference Optimization Engineer"),
    )
    r = dispatch(p, db_session)
    save_r = dispatch(
        MutationPayload(operation="save_draft", target=MutationTarget(draft_id=str(r.draft["id"])), changes=ApplicationChanges()),
        db_session,
    )
    app_dict = save_r.application
    assert app_dict is not None
    assert app_dict["company"] == "Acme"
    assert app_dict["role"] == "LLM Inference Optimization Engineer"
    assert "roles" not in app_dict
    assert "roles_json" not in app_dict


def test_open_ended_role_accepted_unchanged(db_session: Session):
    p = MutationPayload(
        operation="create_draft",
        target=MutationTarget(),
        changes=ApplicationChanges(company="Acme", role="Conversational AI Systems Engineer"),
    )
    r = dispatch(p, db_session)
    assert r.success
    assert r.draft["role"] == "Conversational AI Systems Engineer"


def test_company_patch_updates_company_id(db_session: Session):
    """Patching company on an application must update company_id, not create a stray text column."""
    p = MutationPayload(
        operation="create_draft",
        target=MutationTarget(),
        changes=ApplicationChanges(company="Acme", role="AI Engineer"),
    )
    r = dispatch(p, db_session)
    save_r = dispatch(
        MutationPayload(operation="save_draft", target=MutationTarget(draft_id=str(r.draft["id"])), changes=ApplicationChanges()),
        db_session,
    )
    app_id = save_r.application["id"]

    patch_r = dispatch(
        MutationPayload(
            operation="patch_application",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(company="NewCo"),
        ),
        db_session,
    )
    assert patch_r.success
    assert patch_r.application["company"] == "NewCo"

    companies = db_session.query(Company).all()
    company_names = {c.name for c in companies}
    assert "Acme" in company_names
    assert "NewCo" in company_names
