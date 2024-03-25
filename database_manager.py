import json
from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime, Float, Integer, inspect, Boolean
import pandas as pd
from sqlalchemy.sql import text
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

class DatabaseManager():
    def __init__(self, username, password, host, db_name=None, table_name=None):
        self.username = username
        self.password = password
        self.host = host
        self.db_name = db_name
        self.table_name = table_name
        if db_name:
            self.create_database_if_not_exists()
            self.engine = create_engine('postgresql+psycopg2://{}:{}@{}:5432/{}'.format(username, password, host, db_name))
            self.conn = self.engine.connect()
            self.meta = MetaData()
            self.meta.bind = self.engine
        else:
            self.engine = None
            self.conn = None
            self.meta = None

    def create_database_if_not_exists(self):
        conn = psycopg2.connect(
            dbname="postgres",
            user=self.username,
            password=self.password,
            host=self.host
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{self.db_name}';")
        exists = cur.fetchone()
        if not exists:
            cur.execute(f"CREATE DATABASE {self.db_name};")
        cur.close()
        conn.close()

    def create_user_registration_table(self):
        if not self.check_table_exists('user_registration'):
            table = Table(
                'user_registration',
                self.meta,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('email', String(255), unique=True, nullable=False),
                Column('password', String(255), nullable=False),
                Column('email_confirmed', Boolean, default=False),
            )
            self.meta.create_all(self.engine)
            print("Table 'user_registration' created.")

    def check_table_exists(self, table_name):
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()

    def save_dataframe_to_sql(self, df):
        df.to_sql(self.table_name, self.engine, if_exists='append', index=False)

    def print_top_rows(self, n=5):
        # Print the top n rows of the table
        query = text(f"SELECT * FROM {self.table_name} LIMIT {n}")
        result = self.conn.execute(query).fetchall()
        for row in result:
            print(row)

    def add_column_to_table(self, column_name, column_type):
        try:
            with self.engine.connect() as connection:
                connection.execute(text(f'ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS {column_name} {column_type}'))
            print(f"Successfully added column {column_name} to the table {self.table_name}.")
        except SQLAlchemyError as e:
            print(f"Error occurred while adding column: {e}")

    def select_row_by_id(self, table_name, id):
        query = text(f"SELECT * FROM {table_name} WHERE id = {id}")
        result = self.conn.execute(query)
        return result.fetchall()

    def save_to_page_embeddings(self, data):
        parsed_pages = Table('parsed_pages_2', self.meta, autoload_with=self.engine)

        with self.conn.begin() as trans:
            try:
                insert_stmt = parsed_pages.insert().values(
                    paper_id=data['paper_id'],
                    url=data['url'],
                    page_number=data['page_number'],
                    content=data['content'],
                    page_embeddings=data['embeddings']
                )
                self.conn.execute(insert_stmt)
                logging.info(f"Successfully inserted page {data['page_number']} for paper ID {data['paper_id']}")
            except IntegrityError:
                logging.info(f"Page {data['page_number']} for paper ID {data['paper_id']} already exists. Skipping.")
            except Exception as e:
                logging.info(f"Could not insert page {data['page_number']} for paper ID {data['paper_id']}. Error: {str(e)}")


    def fetch_embeddings(self, table_name):
        table = Table(table_name, self.meta, autoload_with=self.engine)
        # Assuming page_embeddings is the 7th column (change index accordingly if different)
        return [row[6] for row in self.conn.execute(table.select()).fetchall()]


    def add_new_column(self, column_name, column_type):
        # Add a new column to the table if it does not exist already
        if not self.check_column_exists(column_name):
            query = f"ALTER TABLE {self.table_name} ADD COLUMN {column_name} {column_type};"
            self.conn.execute(query)
            self.conn.commit()
            print(f"Column '{column_name}' added to the table '{self.table_name}'.")
        else:
            print(f"Column '{column_name}' already exists in the table '{self.table_name}'.")

    def check_column_exists(self, column_name):
        inspector = inspect(self.engine)
        columns = [column['name'] for column in inspector.get_columns(self.table_name)]
        return column_name in columns


    def add_missing_columns(self, data):
        # Create a connection from the engine
        with self.engine.connect() as connection:
            # Get the existing columns in the table
            existing_columns = connection.execute(text(f"PRAGMA table_info({self.table_name})")).fetchall()

        # Extract the column names from the existing_columns list
        existing_column_names = [column[1] for column in existing_columns]

        # Iterate through the columns in the DataFrame and add them to the SQLite table if they don't already exist
        for column in data.columns:
            if column not in existing_column_names:
                with self.engine.connect() as connection:
                    connection.execute(text(f'ALTER TABLE {self.table_name} ADD COLUMN "{column}" REAL'))


    def drop_table(self, table_name):
        try:
            with self.engine.connect() as connection:
                connection.execute(text(f'DROP TABLE IF EXISTS {table_name} CASCADE'))
                connection.commit()  # manually commit the transaction
            print(f"Successfully dropped table {table_name}.")
        except SQLAlchemyError as e:
            print(f"Error occurred while dropping table: {e}")



    def fetch_table_names(self):
        # Fetch all table names in the database
        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()
        return table_names

    def create_table_if_not_exists(self, df):
        if not self.check_table_exists(self.table_name):
            # Define a new table with a name, columns and their types
            table = Table(
                self.table_name,
                self.meta,
                Column('user_id', String, nullable=False),
                Column('strategy_id', String, nullable=False),
                Column('strategy_type', String, nullable=False),
                Column('target_period', Integer, nullable=False),
                Column('stock_universe', String, nullable=False),
            )
            # Create the table
            self.meta.create_all(self.engine)
            print(f"Table '{self.table_name}' created.")

    def clone_table(self, source_db_manager, source_table_name, target_table_name, user_id):
        # Fetch rows from source table
        source_rows = source_db_manager.conn.execute(f"SELECT * FROM {source_table_name}")

        # Create target table if not exists
        self.create_table_if_not_exists_from_source(source_rows.keys(), target_table_name)

        # Add user_id to each row and insert it into the target table
        for row in source_rows:
            data_with_user_id = dict(row)
            data_with_user_id['user_id'] = user_id
            self.conn.execute(f"INSERT INTO {target_table_name} VALUES (:data)", {'data': data_with_user_id})

    def create_table_if_not_exists_from_source(self, column_names, target_table_name):
        if not self.check_table_exists(target_table_name):
            table = Table(target_table_name, self.meta, *[Column(name, String, nullable=False) for name in column_names], Column('user_id', String, nullable=False))
            self.meta.create_all(self.engine)
            print(f"Table '{target_table_name}' created.")

    def update_distance(self, table_name, id, distance):
        with self.engine.connect() as connection:
            connection.execute(text(f"UPDATE {table_name} SET distance = {distance} WHERE id = {id}"))
        print(f"Successfully updated distance for ID: {id}")


    def update_embedding(self, table_name, id, embeddings):
        with self.engine.begin() as connection:
            query = text(
                f"""UPDATE {table_name} 
                   SET embeddings = :embeddings 
                   WHERE id = :id"""
            )
            connection.execute(query, {"id": id, "embeddings": embeddings.tolist()})

    def fetch_all_databases(self):
        conn = psycopg2.connect(
            dbname="postgres",
            user=self.username,
            password=self.password,
            host=self.host
        )
        cur = conn.cursor()
        cur.execute("SELECT datname FROM pg_database;")
        db_names = cur.fetchall()
        cur.close()
        conn.close()
        return [db_name[0] for db_name in db_names]


    def fetch_columns(self, table_name):
        if self.check_table_exists(table_name):
            inspector = inspect(self.engine)
            return [column['name'] for column in inspector.get_columns(table_name)]
        else:
            return None


    def fetch_rows(self, table_name, filter_condition=None):
        query = "SELECT * FROM " + table_name
        if filter_condition:
            query += " WHERE " + filter_condition
        result = self.conn.execute(text(query))
        return result.fetchall()

    def delete_rows(self, table_name, condition):
        with self.engine.begin() as connection:
            query = text(f"DELETE FROM {table_name} WHERE {condition}")
            connection.execute(query)


    def insert_features_to_db(df, feature_list, db_manager, feature_set_id):

        feature_set_data = Table('feature_set_data', db_manager.meta, autoload_with=db_manager.engine)

        # Loop through each feature in the feature_list
        for feature in feature_list:
            for index, row in df.iterrows():
                # Insert the feature name, date, and value into the feature_set_data table
                stmt = feature_set_data.insert().values(
                    feature_set_id=feature_set_id,
                    feature_name=feature,
                    date=row['date'], 
                    value=row[feature]
                )
                db_manager.conn.execute(stmt)



    def create_technical_indicators_table(self):
        if not self.check_table_exists('technical_indicators'):
            table = Table(
                'technical_indicators',
                self.meta,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('user_id', String(255), nullable=False),
                Column('feature_name', String(255), nullable=False),
                Column('feature_definition', String(1000), nullable=False),
                Column('formula', String(1000), nullable=False),
                Column('parameters', String(1000), nullable=False),
            )
            self.meta.create_all(self.engine)
            print("Table 'technical_indicators' created.")



    def get_all_technical_indicators(self):
        if not self.check_table_exists('technical_indicators'):
            print("Table 'technical_indicators' does not exist.")
            return None

        print("Executing query: SELECT * FROM technical_indicators")
        query = text(f"SELECT * FROM technical_indicators")
        result = self.conn.execute(query).fetchall()

        print(f"Query result: {result}")

        # Print each row
        for row in result:
            print(row)

        if result:
            # Get column names using Inspector
            inspector = inspect(self.conn)
            keys = [column['name'] for column in inspector.get_columns('technical_indicators')]
            print(f"Query result keys: {keys}")

            indicators = []
            for row in result:
                row_dict = dict(zip(keys, row))    # Map the data for each row to its column name

                # Debug print here
                print("Type and value of 'parameters':", type(row_dict.get('parameters')), row_dict.get('parameters'))

                # Check if 'parameters' is already a dict
                if isinstance(row_dict['parameters'], dict):
                    parameters = row_dict['parameters']
                else:
                    try:
                        parameters = json.loads(row_dict['parameters']) if row_dict['parameters'] else {}
                    except json.JSONDecodeError:
                        parameters = {}

                row_dict.update(parameters=parameters)
                indicators.append(row_dict)
            return indicators





    def get_user_technical_indicators(self, user_id):
        if not self.check_table_exists('technical_indicators'):
            print("Table 'technical_indicators' does not exist.")
            return []

        print(f"Executing query: SELECT DISTINCT ON (feature_name) * FROM technical_indicators WHERE user_id = {user_id} ORDER BY feature_name")
        query = text(f"SELECT DISTINCT ON (feature_name) * FROM technical_indicators WHERE user_id = :user_id ORDER BY feature_name")
        result = self.conn.execute(query, {"user_id": user_id}).fetchall()

        print(f"Query result for user_id {user_id}: {result}")

        # Print each row
        for row in result:
            print(row)

        if result:
            try:
                # Get column names using Inspector (as in 'get_all_technical_indicators')
                inspector = inspect(self.conn)
                keys = [column['name'] for column in inspector.get_columns('technical_indicators')]
                print(f"Query result keys for user_id {user_id}: {keys}")

                indicators = []
                for row in result:
                    row_dict = dict(zip(keys, row))
                    print(f"Row dict: {row_dict}")  # Add this line

                    try:
                        parameters = json.loads(row_dict['parameters']) if row_dict['parameters'] else {}
                    except json.JSONDecodeError:
                        parameters = {}

                    print(f"Parameters: {parameters}")

                    row_dict.update(parameters=parameters)
                    indicators.append(row_dict)
                return indicators
            except IndexError:
                print(f"No technical indicators found for user_id: {user_id}")
                return []
        else:
            print(f"No technical indicators found for user_id: {user_id}")
            return []


    def insert_user_technical_indicators(self, user_id, indicator):
        self.create_technical_indicators_table()

        params_json = json.dumps(indicator['parameters']) if indicator['parameters'] else '{}'

        # Check if indicator already exists
        check_query = text("SELECT * FROM technical_indicators WHERE user_id = :user_id AND feature_name = :feature_name")
        existing_indicators = self.conn.execute(check_query, {"user_id": user_id, "feature_name": indicator['feature_name']}).fetchall()
        
        if existing_indicators:
            print(f"Indicator with feature_name {indicator['feature_name']} already exists for user_id {user_id}. Skipping insertion.")
            return

        print(f"Preparing to insert indicator for user_id {user_id}: {indicator}")
        query = text("INSERT INTO technical_indicators (user_id, feature_name, feature_definition, formula, parameters) VALUES (:user_id, :feature_name, :feature_definition, :formula, :parameters)")

        print(f"Executing query: {query}, Parameters: user_id={user_id}, feature_name={indicator['feature_name']}, feature_definition={indicator['feature_definition']}, formula={indicator['formula']}, parameters={params_json}")

        result = None
        try:
            result = self.conn.execute(query, {"user_id": user_id, "feature_name": indicator['feature_name'], "feature_definition": indicator['feature_definition'], "formula": indicator['formula'], "parameters": params_json})
        except SQLAlchemyError as e:
            print(f"Error: {str(e)}")
            self.conn.rollback()
            raise e
        else:
            self.conn.commit()
            print(f"Inserted technical indicator for user_id {user_id}: {indicator}")




