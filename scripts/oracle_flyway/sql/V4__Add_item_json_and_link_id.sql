-- V4: Add ITEM_JSON column to CONVERSATION_ITEMS and LINK_ID column +
-- monotonic sequence to CONVERSATION_ITEM_LINKS.
--
-- ITEM_JSON (CLOB with IS JSON check) holds the whole canonical OpenAI
-- item payload so reads can return the correct shape without per-type
-- reconstruction. LINK_ID (NUMBER, populated via CONV_ITEM_LINK_ID_SEQ)
-- is a strictly increasing per-link key that gives /v1/conversations/{id}/items
-- deterministic ordering (replacing ADDED_AT, ITEM_ID which is not unique).
--
-- Mirrors crates/data_connector/src/oracle_migrations.rs::oracle_v12_up.

------------------------------------------------------------
-- 1. CONVERSATION_ITEMS: add ITEM_JSON
------------------------------------------------------------
ALTER TABLE CONVERSATION_ITEMS ADD (ITEM_JSON CLOB CHECK (ITEM_JSON IS JSON));

------------------------------------------------------------
-- 2. CONVERSATION_ITEM_LINKS: add LINK_ID + sequence + unique index
------------------------------------------------------------
-- ORDER guarantees strictly monotonic NEXTVAL across RAC / multi-instance
-- deployments; no-op on single-instance Oracle. Required because LINK_ID
-- backs deterministic conversation-item ordering.
CREATE SEQUENCE CONV_ITEM_LINK_ID_SEQ START WITH 1 INCREMENT BY 1 NOCACHE ORDER;

ALTER TABLE CONVERSATION_ITEM_LINKS ADD (LINK_ID NUMBER);

CREATE UNIQUE INDEX IDX_CONV_LINK_ID ON CONVERSATION_ITEM_LINKS (CONVERSATION_ID, LINK_ID);
