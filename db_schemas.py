# Initialize the database manager
db_manager = DatabaseManager(
    username="postgres",
    password="",
    host="internaldb.c0rz9kyyn4jp.us-east-2.rds.amazonaws.com",
    db_name="internaldb"
)

# Get the engine
engine = db_manager.engine

metadata = MetaData()


raw_papers = Table('raw_papers', metadata,
    Column('paper_id', Integer, primary_key=True),
    Column('title', Text),
    Column('url', Text),
    Column('citation', Integer),
    Column('pdf_or_html_link', Text),
    Column('text', Text),
    Column('abstract', Text),
    Column('abstract_embeddings', ARRAY(Float)),
)

parsed_pages2 = Table('parsed_pages_2', metadata,
    Column('id', Integer, autoincrement=True),  # This column can serve as an additional unique identifier for each entry
    Column('paper_id', Integer, ForeignKey('raw_papers.paper_id')),  # Foreign key referencing raw_papers
    Column('url', Text),
    Column('page_number', Integer),
    Column('content', Text),
    Column('page_embeddings', ARRAY(Float)),
    PrimaryKeyConstraint('paper_id', 'page_number')
)


# Initialize the database manager
db_manager = DatabaseManager(
    username="postgres",
    password="",
    host="internaldb.c0rz9kyyn4jp.us-east-2.rds.amazonaws.com",
    db_name="user_db"
)

# Get the engine
engine = db_manager.engine

metadata = MetaData()



user_query = Table('user_query', metadata,
    Column('id', Integer, primary_key=True),
    Column('user_id', Text, ForeignKey('users.user_id')),  # Assuming 'users' is the table containing user information
    Column('query', Text),
    Column('query_embeddings', ARRAY(Float)),
    Column('abstract_to_query_distance', JSON),
    Column('selected_paper_ids', JSON),
    Column('paper_id_page_number_to_query_distance', JSON),
    Column('closest_paper_id_page_number', JSON),
)



page_features = Table('page_features', metadata,
    Column('id', Integer, primary_key=True),
    Column('user_id', Text, ForeignKey('users.id')),  # Assuming 'users' is the users table
    Column('paper_id', Integer, ForeignKey('raw_papers.paper_id')),
    Column('page_number', Integer, ForeignKey('parsed_pages_2.page_number')),
    Column('feature_name', Text),
    Column('feature_definition', Text),
    Column('feature_formula', Text),
    Column('feature_parameters', JSON),
    Column('feature_selected_for_code_gen', Boolean),
    Column('init_code_generated_from_openAI', JSON),
    Column('method_code_generated_from_openAI', JSON),
    UniqueConstraint('user_id', 'paper_id', 'page_number', 'feature_name'),  # Ensures uniqueness for each combination
)