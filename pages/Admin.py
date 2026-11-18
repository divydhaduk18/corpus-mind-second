"""Administrative Control Center for Corpus Mind (Improvements 28, 29, 31, 33).

Includes:
- Complete Role-Based Access Control (Admin, Editor, Viewer) management
- Account status management (active / deactivated) and password resets
- Global document permissions oversight and ownership transfer
- Real-time Security Audit Log Dashboard with dynamic filters & CSV export
- Dual Database engine diagnostics (SQLite / PostgreSQL)
"""

import streamlit as st
import os
import sys
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.auth import AuthManager
from utils.database import db
from utils.audit import (
    AuditLogger,
    ACTION_USER_CREATE, ACTION_USER_ROLE_CHANGE, ACTION_USER_STATUS_CHANGE,
    ACTION_USER_DELETE, ACTION_DOC_OWNERSHIP_TRANSFER, ACTION_DOC_PERM_GRANT,
    ACTION_DOC_PERM_REVOKE, ACTION_DOC_DELETE
)

# Guard Page Access
AuthManager.require_auth()

# Admin authorization check
if st.session_state.get("user_role") != getattr(config, "ROLE_ADMIN", "admin"):
    st.error("🚫 Access Denied. You do not have permission to view the Admin panel.")
    st.info("Log in with an administrator account to access this page.")
    st.stop()

# Page Styling
css_path = config.CSS_DIR / "style.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar layout
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/robot-3.png", width=90)
    st.markdown(f"### Admin: **{st.session_state.username}**")
    st.markdown(
        '<span style="background-color:#ef4444; color:white; padding:4px 10px; border-radius:12px; font-weight:bold; font-size:12px;">👑 ADMINISTRATOR AREA</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        AuthManager.logout_user()
        st.rerun()

st.title("🛡️ Enterprise Administration & Security Portal")
st.markdown("Manage user access control, document permissions, audit trails, and multi-engine database operations.")

tab_users, tab_perms, tab_audit, tab_docs, tab_db = st.tabs([
    "👥 User Management & RBAC",
    "🔐 Document Permissions",
    "📜 Security Audit Logs",
    "📄 Global Documents",
    "🗄️ Database & Engine"
])

# -----------------
# TAB 1: USER MANAGEMENT & RBAC
# -----------------
with tab_users:
    st.subheader("Create a New User Account")
    
    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        new_username = st.text_input("Username", key="admin_new_user")
    with col_u2:
        new_password = st.text_input("Password", type="password", key="admin_new_pass")
    with col_u3:
        new_role = st.selectbox(
            "Role",
            options=[config.ROLE_ADMIN, config.ROLE_EDITOR, config.ROLE_VIEWER],
            index=1,
            key="admin_new_role",
            help="Admin: full access. Editor: upload & manage own documents. Viewer: read-only access to shared documents."
        )
        
    if st.button("👥 Create User", type="primary", use_container_width=True):
        if not new_username or not new_password:
            st.error("Please provide both a username and password.")
        else:
            success, msg = AuthManager.register_user(new_username, new_password, new_role)
            if success:
                AuditLogger.log(
                    action=ACTION_USER_CREATE,
                    user_id=st.session_state.user_id,
                    username=st.session_state.username,
                    resource_type="user",
                    resource_id=new_username,
                    status="SUCCESS",
                    details={"assigned_role": new_role}
                )
                st.success(f"User '{new_username}' created successfully with role '{new_role}'!")
                st.rerun()
            else:
                st.error(msg)
                
    st.markdown("---")
    st.subheader("Manage Existing Users")
    
    users = db.get_all_users()
    if users:
        df_users = pd.DataFrame(users)
        st.dataframe(df_users, use_container_width=True)
        
        # User Action Form
        st.markdown("#### Modify User Role or Status")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            selected_username = st.selectbox(
                "Select User",
                options=[u["username"] for u in users if u["username"] != st.session_state.username],
                key="admin_select_user"
            )
        
        target_user = db.get_user_by_username(selected_username) if selected_username else None
        
        if target_user:
            with col_m2:
                updated_role = st.selectbox(
                    "Change Role",
                    options=[config.ROLE_ADMIN, config.ROLE_EDITOR, config.ROLE_VIEWER],
                    index=[config.ROLE_ADMIN, config.ROLE_EDITOR, config.ROLE_VIEWER].index(target_user.get("role", config.ROLE_EDITOR)) if target_user.get("role") in [config.ROLE_ADMIN, config.ROLE_EDITOR, config.ROLE_VIEWER] else 1,
                    key="admin_change_role"
                )
            with col_m3:
                current_status = target_user.get("status", "active")
                updated_status = st.selectbox(
                    "Account Status",
                    options=["active", "deactivated"],
                    index=0 if current_status == "active" else 1,
                    key="admin_change_status"
                )
                
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                if st.button("💾 Update Role & Status", use_container_width=True):
                    db.update_user_role(target_user["id"], updated_role)
                    db.update_user_status(target_user["id"], updated_status)
                    AuditLogger.log(
                        action=ACTION_USER_ROLE_CHANGE,
                        user_id=st.session_state.user_id,
                        username=st.session_state.username,
                        resource_type="user",
                        resource_id=target_user["id"],
                        status="SUCCESS",
                        details={"new_role": updated_role, "new_status": updated_status}
                    )
                    st.success(f"User '{selected_username}' updated: Role={updated_role}, Status={updated_status}.")
                    st.rerun()
            with col_b2:
                reset_pw = st.text_input("New Password", type="password", key="admin_reset_pw", placeholder="Type new password...")
                if st.button("🔑 Reset Password", use_container_width=True):
                    if reset_pw:
                        new_h = AuthManager.hash_password(reset_pw)
                        db.update_user_password(target_user["id"], new_h)
                        st.success(f"Password reset for '{selected_username}'.")
                    else:
                        st.warning("Enter a new password first.")
            with col_b3:
                if st.button("🗑️ Delete User Permanently", type="secondary", use_container_width=True):
                    db.delete_user(target_user["id"])
                    AuditLogger.log(
                        action=ACTION_USER_DELETE,
                        user_id=st.session_state.user_id,
                        username=st.session_state.username,
                        resource_type="user",
                        resource_id=target_user["id"],
                        status="SUCCESS"
                    )
                    st.success(f"User '{selected_username}' has been deleted.")
                    st.rerun()
    else:
        st.info("No registered users found.")

# -----------------
# TAB 2: DOCUMENT PERMISSIONS
# -----------------
with tab_perms:
    st.subheader("🔐 Document Sharing & Access Control Inspector")
    st.markdown("Inspect and manage document-level permissions (Owner, Editor, Viewer).")
    
    all_docs = db.get_all_documents()
    if all_docs:
        doc_options = {f"#{d['id']}: {d['file_name']} (Owner: {d.get('owner_username', 'Unknown')})": d['id'] for d in all_docs}
        selected_doc_label = st.selectbox("Select Document", options=list(doc_options.keys()), key="perm_select_doc")
        selected_doc_id = doc_options[selected_doc_label]
        
        current_perms = db.get_document_permissions(selected_doc_id)
        st.markdown(f"**Current Permissions for Document #{selected_doc_id}:**")
        if current_perms:
            df_perms = pd.DataFrame(current_perms)
            st.dataframe(df_perms[["id", "user_name", "permission_level", "granted_by_name", "created_at"]], use_container_width=True)
        else:
            st.info("This document has no external permissions granted (only accessible by its primary owner and administrators).")
            
        st.markdown("---")
        st.markdown("#### Grant or Update Access Permission")
        all_system_users = db.get_all_users()
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            grantee_user = st.selectbox(
                "Grant Access To User",
                options=[u["username"] for u in all_system_users],
                key="perm_grant_user"
            )
        with col_p2:
            grant_level = st.selectbox(
                "Permission Level",
                options=["viewer", "editor", "none"],
                index=0,
                key="perm_grant_level",
                help="Viewer: can search and chat. Editor: can edit metadata. None: revokes access."
            )
        with col_p3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("Apply Permission", type="primary", use_container_width=True):
                target_u = db.get_user_by_username(grantee_user)
                if target_u:
                    db.grant_document_permission(
                        document_id=selected_doc_id,
                        user_id=target_u["id"],
                        permission_level=grant_level,
                        granted_by=st.session_state.user_id
                    )
                    AuditLogger.log(
                        action=ACTION_DOC_PERM_GRANT if grant_level != "none" else ACTION_DOC_PERM_REVOKE,
                        user_id=st.session_state.user_id,
                        username=st.session_state.username,
                        resource_type="document",
                        resource_id=selected_doc_id,
                        status="SUCCESS",
                        details={"grantee_id": target_u["id"], "grantee": grantee_user, "permission": grant_level}
                    )
                    st.success(f"Permission '{grant_level}' set for '{grantee_user}'.")
                    st.rerun()

        st.markdown("---")
        st.markdown("#### Transfer Primary Ownership")
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            new_owner_name = st.selectbox(
                "Transfer Primary Ownership To",
                options=[u["username"] for u in all_system_users],
                key="perm_transfer_owner"
            )
        with col_t2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("Transfer Ownership", type="secondary", use_container_width=True):
                new_owner = db.get_user_by_username(new_owner_name)
                if new_owner:
                    success = db.transfer_document_ownership(
                        document_id=selected_doc_id,
                        new_owner_id=new_owner["id"],
                        transferred_by=st.session_state.user_id
                    )
                    if success:
                        AuditLogger.log(
                            action=ACTION_DOC_OWNERSHIP_TRANSFER,
                            user_id=st.session_state.user_id,
                            username=st.session_state.username,
                            resource_type="document",
                            resource_id=selected_doc_id,
                            status="SUCCESS",
                            details={"new_owner_id": new_owner["id"], "new_owner": new_owner_name}
                        )
                        st.success(f"Ownership of document #{selected_doc_id} transferred to '{new_owner_name}'.")
                        st.rerun()
    else:
        st.info("No documents found in system.")

# -----------------
# TAB 3: SECURITY AUDIT LOGS
# -----------------
with tab_audit:
    st.subheader("📜 Security & System Audit Trail")
    st.markdown("Review persistent, tamper-evident security audit logs. Sensitive passwords, API keys, and tokens are redacted.")
    
    # Filter Controls
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        filter_action = st.text_input("Filter by Action", placeholder="e.g. AUTH, DOC, USER, RAG")
    with col_f2:
        filter_status = st.selectbox("Status", options=["All", "SUCCESS", "FAILURE", "DENIED"], index=0)
    with col_f3:
        filter_resource = st.selectbox("Resource", options=["All", "user", "document", "rag", "system"], index=0)
    with col_f4:
        limit_logs = st.selectbox("Max Records", options=[50, 100, 250, 500], index=1)
        
    resolved_status = None if filter_status == "All" else filter_status
    resolved_resource = None if filter_resource == "All" else filter_resource
    
    logs = AuditLogger.get_logs(
        action=filter_action if filter_action.strip() else None,
        status=resolved_status,
        resource_type=resolved_resource,
        limit=limit_logs
    )
    
    total_matching = AuditLogger.get_count(
        action=filter_action if filter_action.strip() else None,
        status=resolved_status,
        resource_type=resolved_resource
    )
    
    st.write(f"Displaying **{len(logs)}** of **{total_matching}** matching audit entries.")
    
    if logs:
        # Export CSV Button
        csv_data = AuditLogger.export_to_csv(logs)
        st.download_button(
            label="📥 Export Filtered Audit Logs as CSV",
            data=csv_data,
            file_name=f"audit_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        df_audit = pd.DataFrame(logs)
        st.dataframe(
            df_audit[["id", "timestamp", "username", "action", "resource_type", "resource_id", "status", "ip_address", "details_json"]],
            use_container_width=True
        )
    else:
        st.info("No audit logs match current filter criteria.")

# -----------------
# TAB 4: GLOBAL DOCUMENTS
# -----------------
with tab_docs:
    st.subheader("Global Document Index")
    st.markdown("View and manage all uploaded documents across all system users.")
    
    global_docs = db.get_all_documents()
    if global_docs:
        df_docs = pd.DataFrame(global_docs)
        st.dataframe(df_docs, use_container_width=True)
        
        # Select document to delete
        doc_to_delete = st.selectbox(
            "Select Document to Delete Globally",
            options=list(df_docs["file_name"]),
            key="admin_del_doc"
        )
        
        if st.button("🗑️ Delete Selected File Globally", use_container_width=True):
            for doc in global_docs:
                if doc['file_name'] == doc_to_delete:
                    db.delete_document(doc['id'])
                    AuditLogger.log(
                        action=ACTION_DOC_DELETE,
                        user_id=st.session_state.user_id,
                        username=st.session_state.username,
                        resource_type="document",
                        resource_id=doc['id'],
                        status="SUCCESS",
                        details={"file_name": doc_to_delete, "scope": "global_admin"}
                    )
                    from utils.rag import RAGManager
                    RAGManager.rebuild_user_index(doc['user_id'])
                    st.success(f"Document '{doc_to_delete}' deleted and user {doc['user_id']} vector index updated.")
                    st.rerun()
    else:
        st.info("No uploaded documents indexed in system.")

# -----------------
# TAB 5: DATABASE & ENGINE DIAGNOSTICS
# -----------------
with tab_db:
    st.subheader("Relational Database & Engine Status")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.metric("Database Engine", "PostgreSQL" if db.is_postgres else "SQLite")
        st.write(f"**Database URL / Path:** `{db.db_url}`")
    with col_d2:
        st.metric("Pool Size", getattr(config, "DB_POOL_SIZE", 10))
        st.write(f"**Max Overflow:** `{getattr(config, 'DB_MAX_OVERFLOW', 20)}`")
        
    st.markdown("---")
    st.markdown("#### Table Record Counts")
    table_names = [
        "users", "documents", "document_permissions", "chat_sessions",
        "messages", "document_jobs", "ragas_evaluations", "audit_logs", "analytics_logs"
    ]
    
    counts = {}
    for tbl in table_names:
        try:
            res = db._execute(f"SELECT COUNT(*) as cnt FROM {tbl}", fetch_one=True)
            counts[tbl] = res["cnt"] if res else 0
        except Exception:
            counts[tbl] = "N/A"
            
    df_counts = pd.DataFrame([{"Table Name": k, "Record Count": v} for k, v in counts.items()])
    st.table(df_counts)
