import streamlit as st
from sqlalchemy import create_engine, inspect
import pandas as pd
from prettytable import PrettyTable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import Ollama

def get_table_schema(engine, table_name: str) -> PrettyTable:
    """Retrieve and format the schema of a given table."""
    inspector = inspect(engine)
    schema_table = PrettyTable()
    schema_table.field_names = ["Column Name", "Data Type"]

    columns = inspector.get_columns(table_name)
    for column in columns:
        schema_table.add_row([column['name'], column['type']])
    
    return schema_table

def get_sample_data(engine, table_name: str, sample_size: int = 1) -> PrettyTable:
    """Retrieve and format sample data from a given table."""
    query = f'SELECT * FROM "{table_name}" LIMIT {sample_size};'
    df = pd.read_sql(query, engine)
    
    sample_data_table = PrettyTable()
    if not df.empty:
        sample_data_table.field_names = df.columns.tolist()
        for _, row in df.iterrows():
            sample_data_table.add_row(row.tolist())
    
    return sample_data_table

def display_table_info(db_uri: str, sample_size: int = 1) -> None:
    """Display schema and sample data for all tables in the database."""
    engine = create_engine(db_uri)
    inspector = inspect(engine)
    
    for table_name in inspector.get_table_names():
        print(f"\nTable: {table_name}\nSchema:")
        print(get_table_schema(engine, table_name))
        print("\nSample Data:")
        print(get_sample_data(engine, table_name, sample_size))

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    """Initialize and return a connection to the database."""
    db_uri = uri_database(user, password, host, port, database)
    return SQLDatabase.from_uri(db_uri)

def uri_database(user: str, password: str, host: str, port: str, database: str) -> str:
    """Create and return the database URI."""
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

def get_sql_chain(db: SQLDatabase):
    """Create and return an SQL query chain based on the database schema."""
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    Do not reply to the user, and only respond with SQL queries.
    
    For example:
    Question: which 3 genres have the most tracks?
    SQL Query: SELECT GenreId, COUNT(*) as track_count FROM "Track" GROUP BY GenreId ORDER BY track_count DESC LIMIT 3;

    Question: Name 10 playlists.
    SQL Query: SELECT "Name" FROM "Playlist" LIMIT 10;

    Question: What are the 5 most recent invoices?
    SQL Query: SELECT * FROM "Invoice" ORDER BY "InvoiceDate" DESC LIMIT 5;

    Question: List the names and titles of employees and their managers.
    SQL Query: SELECT e1."FirstName" AS EmployeeFirstName, e1."LastName" AS EmployeeLastName, e1."Title" AS EmployeeTitle, e2."FirstName" AS ManagerFirstName, e2."LastName" AS ManagerLastName, e2."Title" AS ManagerTitle FROM "Employee" e1 LEFT JOIN "Employee" e2 ON e1."ReportsTo" = e2."EmployeeId";

    Question: What is the average unit price of tracks by genre?
    SQL Query: SELECT g."Name" AS GenreName, AVG(t."UnitPrice") AS AverageUnitPrice FROM "Track" t JOIN "Genre" g ON t."GenreId" = g."GenreId" GROUP BY g."Name" ORDER BY AverageUnitPrice DESC;

    Question: How many albums does each artist have?
    SQL Query: SELECT a."ArtistId", a."Name" AS ArtistName, COUNT(al."AlbumId") AS AlbumCount FROM "Artist" a JOIN "Album" al ON a."ArtistId" = al."ArtistId" GROUP BY a."ArtistId", a."Name" ORDER BY AlbumCount DESC;

    Question: List all customers from Canada.
    SQL Query: SELECT "CustomerId", "FirstName", "LastName", "Email" FROM "Customer" WHERE "Country" = 'Canada';

    Question: What is the total sales for each customer?
    SQL Query: SELECT c."CustomerId", c."FirstName", c."LastName", SUM(i."Total") AS TotalSales FROM "Customer" c JOIN "Invoice" i ON c."CustomerId" = i."CustomerId" GROUP BY c."CustomerId", c."FirstName", c."LastName" ORDER BY TotalSales DESC;

    Question: List the names of tracks that are longer than 5 minutes.
    SQL Query: SELECT "Name" FROM "Track" WHERE "Milliseconds" > 300000;

    Question: List the titles of albums by AC/DC.
    SQL Query: SELECT al."Title" FROM "Album" al JOIN "Artist" a ON al."ArtistId" = a."ArtistId" WHERE a."Name" = 'AC/DC';
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = Ollama(model="llama3:instruct", base_url="http://ollama.ollama.svc.cluster.local:11434", verbose=True)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    """Generate a response based on the user's query and the database schema."""
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
    prompt = ChatPromptTemplate.from_template(template)
    llm = Ollama(model="llama3:instruct", base_url="http://ollama.ollama.svc.cluster.local:11434", verbose=True)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: print(vars["query"]) or db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
  
    result = chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })
    
    print(f"Generated SQL Query: {result}")
    
    return result

def llama_page():
    st.title("Direct Llama Interaction")

    # Initialize chat history in session state if not already done
    if "llama_chat_history" not in st.session_state:
        st.session_state.llama_chat_history = []

    # Display chat history
    for message in st.session_state.llama_chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    # Input for user query
    user_query = st.chat_input("Type a message to Llama...")
    if user_query is not None and user_query.strip() != "":
        st.session_state.llama_chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        with st.chat_message("AI"):
            llm = Ollama(model="llama3:instruct", base_url="http://ollama.ollama.svc.cluster.local:11434", verbose=True)
            response = llm(user_query)
            st.markdown(response)
            
        st.session_state.llama_chat_history.append(AIMessage(content=response))

def main():
    st.set_page_config(page_title="Database Q&A and Llama Interaction", page_icon=":speech_balloon:")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Database Q&A", "Direct Llama Interaction"])

    if page == "Database Q&A":
        st.title("Ask Database Anything")

        # Sidebar for database connection inputs
        with st.sidebar:
            st.subheader("Connection to Postgres Database")
            st.write("Do ensure that you are connected to the database before asking anything.")
            host = st.text_input("Host", value="postgres.ollama.svc.cluster.local")
            port = st.text_input("Port", value="5432")
            user = st.text_input("User", value="user")
            password = st.text_input("Password", type="password", value="pass")
            database = st.text_input("Database", value="chinook")
            
            if st.button("Connect"):
                with st.spinner("Connecting to database..."):
                    try:
                        db = init_database(user, password, host, port, database)
                        st.session_state.db = db
                        st.success("Connected to database!")
                        display_table_info(uri_database(user, password, host, port, database))
                    except Exception as e:
                        st.error(f"Failed to connect to database: {e}")

        # Initialize chat history in session state if not already done
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello! Feel free to ask me anything about your database after establishing connection."),
            ]

        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)

        # Input for user query
        user_query = st.chat_input("Type a message...")
        if user_query is not None and user_query.strip() != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            
            with st.chat_message("Human"):
                st.markdown(user_query)
                
            with st.chat_message("AI"):
                response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                st.markdown(response)
                
            st.session_state.chat_history.append(AIMessage(content=response))

    elif page == "Direct Llama Interaction":
        llama_page()

if __name__ == "__main__":
    main()
