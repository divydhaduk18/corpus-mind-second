"""001_initial_schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-11-18 10:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. users
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=False, server_default='editor'),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='active'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username')
    )

    # 2. chat_sessions
    op.create_table(
        'chat_sessions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 3. messages
    op.create_table(
        'messages',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('model', sa.String(length=100), nullable=True),
        sa.Column('response_time_ms', sa.Integer(), nullable=True, server_default='0'),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 4. documents
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('file_name', sa.String(length=255), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('file_path', sa.String(length=1024), nullable=False),
        sa.Column('upload_time', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True, server_default='uploaded'),
        sa.Column('chunk_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('doc_hash', sa.String(length=64), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True, server_default='General'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 5. document_permissions
    op.create_table(
        'document_permissions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('permission_level', sa.String(length=20), nullable=False, server_default='viewer'),
        sa.Column('granted_by', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['granted_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id', 'user_id', name='uq_doc_user_perm')
    )

    # 6. document_jobs
    op.create_table(
        'document_jobs',
        sa.Column('job_id', sa.String(length=100), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('file_name', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.String(length=1024), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=True, server_default='queued'),
        sa.Column('progress_pct', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('current_stage', sa.String(length=100), nullable=True, server_default='queued'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('job_id')
    )

    # 7. ragas_evaluations
    op.create_table(
        'ragas_evaluations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('dataset_name', sa.String(length=255), nullable=False),
        sa.Column('provider', sa.String(length=100), nullable=False),
        sa.Column('model', sa.String(length=100), nullable=False),
        sa.Column('embedding_model', sa.String(length=100), nullable=False),
        sa.Column('avg_faithfulness', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('avg_answer_relevancy', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('avg_context_precision', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('avg_context_recall', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('avg_answer_correctness', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('avg_context_utilization', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('metrics_json', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 8. audit_logs
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('username', sa.String(length=100), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('resource_type', sa.String(length=50), nullable=True, server_default='system'),
        sa.Column('resource_id', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True, server_default='SUCCESS'),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('details_json', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )

    # 9. analytics_logs
    op.create_table(
        'analytics_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('response_time_ms', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('model_used', sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('analytics_logs')
    op.drop_table('audit_logs')
    op.drop_table('ragas_evaluations')
    op.drop_table('document_jobs')
    op.drop_table('document_permissions')
    op.drop_table('documents')
    op.drop_table('messages')
    op.drop_table('chat_sessions')
    op.drop_table('users')
