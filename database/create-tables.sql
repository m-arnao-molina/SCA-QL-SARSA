CREATE SEQUENCE executions_id_seq;
CREATE SEQUENCE iterations_id_seq;
CREATE SEQUENCE execution_results_id_seq;

CREATE TABLE executions (
    id INTEGER DEFAULT nextval('executions_id_seq'::regclass) NOT NULL,
    algorithm_code_name CHARACTER VARYING(100),
    parameters TEXT,
    status CHARACTER VARYING(20),
    start_datetime TIMESTAMP(6) WITHOUT TIME ZONE,
    end_datetime TIMESTAMP(6) WITHOUT TIME ZONE,
    PRIMARY KEY (id)
);

CREATE TABLE iterations (
    id BIGINT DEFAULT nextval('iterations_id_seq'::regclass) NOT NULL,
    execution_id BIGINT,
    iteration_number INTEGER,
    best_fitness DOUBLE PRECISION,
    avg_fitness DOUBLE PRECISION,
    fitness_best_iteration DOUBLE PRECISION,
    parameters TEXT,
    start_datetime TIMESTAMP(6) WITHOUT TIME ZONE,
    end_datetime TIMESTAMP(6) WITHOUT TIME ZONE,
    internal_data BYTEA,
    PRIMARY KEY (id),
    CONSTRAINT iteration_execution_fk FOREIGN KEY (execution_id) REFERENCES "executions" ("id")
);

CREATE TABLE execution_results (
    id BIGINT DEFAULT nextval('execution_results_id_seq'::regclass) NOT NULL,
    execution_id BIGINT,
    fitness DOUBLE PRECISION,
    best_solution TEXT,
    start_datetime TIMESTAMP(6) WITHOUT TIME ZONE,
    end_datetime TIMESTAMP(6) WITHOUT TIME ZONE,
    PRIMARY KEY (id),
    CONSTRAINT execution_result_execution_fk FOREIGN KEY (execution_id) REFERENCES "executions" ("id")
);
