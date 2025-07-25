from database_connector.connect_sql import Connector
import sqlalchemy
import json

connector = Connector(path="database_connector/.env")

def generate_column_info(output_path = "datasets/columns.json"):
    inspector = sqlalchemy.inspect(connector.engine)
    tables = inspector.get_table_names()
    columns = {}
    for table in tables:
        columns[table] = inspector.get_columns(table)
        for column in columns[table]:
            for k,v in column.items():
                if k =="type":
                    column[k] = str(v)
    with open(output_path, "w") as f:
        json.dump(columns, f, indent=4)


def generate_table_row_info(output_path = "datasets/tables.json"):
    inspector = sqlalchemy.inspect(connector.engine)
    tables = inspector.get_table_names()
    table_info = {}
    for table in tables:
        table_info[table] = {
            "columns": [col["name"] for col in inspector.get_columns(table)],
            "primary_key": inspector.get_pk_constraint(table)["constrained_columns"],
            "foreign_keys": inspector.get_foreign_keys(table)
        }
    with open(output_path, "w") as f:
        json.dump(table_info, f, indent=4)

def get_fields_of_interest(path = "sciencebenchmark_dataset/cordis/dev.json",natural_language_col = "question", sql_query_col = "query", correct_gt_col = "all_values_found",
                           output_path = "datasets/total_interest.json"):
    with open(path, "r") as f:
        data = json.load(f)
    total_interest = []
    for item in data:
        diction = {}
        diction["query"] = item[sql_query_col]
        diction["question"] = item[natural_language_col]
        diction["is_correct"] = item[correct_gt_col]
        total_interest.append(diction)
    with open(output_path, "w") as f:
        json.dump(total_interest, f)