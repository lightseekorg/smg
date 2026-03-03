-- V2: Create V2 conversation tables and alter RESPONSES
-- CI-only version: tables + indexes, no sweep procedures or scheduler jobs.

------------------------------------------------------------
-- 1. CONVERSATIONS_V2
------------------------------------------------------------
CREATE TABLE CONVERSATIONS_V2 (
    ID                       VARCHAR2(255)              NOT NULL,
    CONVERSATION_STORE_ID    VARCHAR2(255),
    GENERATIVE_AI_PROJECT_ID VARCHAR2(255),
    CREATED_AT               TIMESTAMP WITH TIME ZONE   NOT NULL,
    METADATA                 CLOB,
    SHORT_TERM_MEMORY        CLOB,
    SHORT_TERM_MEMORY_BYTES  NUMBER,
    UPDATED_AT               TIMESTAMP WITH TIME ZONE,
    EXPIRES_AT               TIMESTAMP WITH TIME ZONE   DEFAULT (SYSTIMESTAMP + INTERVAL '30' DAY) NOT NULL,
    CONSTRAINT PK_CONVERSATIONS_V2 PRIMARY KEY (ID)
);

CREATE INDEX IDX_CONV_V2_EXPIRES_AT ON CONVERSATIONS_V2 (EXPIRES_AT);
CREATE INDEX IDX_CONV_V2_PROJECT_ID ON CONVERSATIONS_V2 (GENERATIVE_AI_PROJECT_ID);
CREATE INDEX IDX_CONV_V2_STORE_ID ON CONVERSATIONS_V2 (CONVERSATION_STORE_ID);

------------------------------------------------------------
-- 2. CONVERSATION_ITEMS
------------------------------------------------------------
CREATE TABLE CONVERSATION_ITEMS (
    ID                       VARCHAR2(255)              NOT NULL,
    RESPONSE_ID              VARCHAR2(255),
    ITEM_TYPE                VARCHAR2(100),
    ROLE                     VARCHAR2(50),
    CONTENT                  CLOB,
    STATUS                   VARCHAR2(50),
    CREATED_AT               TIMESTAMP WITH TIME ZONE,
    EXPIRES_AT               TIMESTAMP WITH TIME ZONE   DEFAULT (SYSTIMESTAMP + INTERVAL '30' DAY) NOT NULL,
    CONSTRAINT PK_CONVERSATION_ITEMS PRIMARY KEY (ID)
);

CREATE INDEX IDX_CONV_ITEMS_EXPIRES_AT ON CONVERSATION_ITEMS (EXPIRES_AT);
CREATE INDEX IDX_CONV_ITEMS_RESPONSE_ID ON CONVERSATION_ITEMS (RESPONSE_ID);
CREATE INDEX IDX_CONV_ITEMS_ITEM_TYPE ON CONVERSATION_ITEMS (ITEM_TYPE);

------------------------------------------------------------
-- 3. CONVERSATION_ITEM_LINKS
------------------------------------------------------------
CREATE TABLE CONVERSATION_ITEM_LINKS (
    CONVERSATION_ID          VARCHAR2(255)              NOT NULL,
    ITEM_ID                  VARCHAR2(255)              NOT NULL,
    ADDED_AT                 TIMESTAMP WITH TIME ZONE   NOT NULL,
    CONSTRAINT PK_CONV_ITEM_LINKS PRIMARY KEY (CONVERSATION_ID, ITEM_ID)
);

CREATE INDEX IDX_CONV_ITEM_LINKS_ORDER ON CONVERSATION_ITEM_LINKS (CONVERSATION_ID, ADDED_AT);

------------------------------------------------------------
-- 4. ALTER RESPONSES: add SAFETY_IDENTIFIER
------------------------------------------------------------
ALTER TABLE RESPONSES ADD (SAFETY_IDENTIFIER VARCHAR2(255));
