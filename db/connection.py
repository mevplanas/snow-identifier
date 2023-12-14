import os

from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import pyodbc

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join("..", "mssql-creds.env")


class SQLServerConnection:
    """
    Connection configuration and connection instance to MSSQL.
    """

    def __init__(self) -> None:
        load_dotenv(dotenv_path=os.path.join(CURR_PATH, ENV_FILE))
        # Loading MS SQL Server database credentials
        self.user = os.environ.get("SQLSERVER_USER")
        self.pw = os.environ.get("SQLSERVER_PASSWORD")
        self.host = os.environ.get("SQLSERVER_HOST")
        self.db = os.environ.get("SQLSERVER_DB")
        self.driver = os.environ.get("SQLSERVER_DRIVER")

    def engine(self):
        """
        MSSQL Connection using SQL Alchemy.
        """

        engine = create_engine(
            f"mssql+pyodbc://{self.user}:{self.pw}@{self.host}\
        /{self.db}?driver={self.driver}&NeedODBCTypesOnly=1",
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

        return engine

    def _con_pyodbc(self) -> pyodbc.Connection:
        """
        MSSQL Connection object using PyODBC.
        """

        conn = pyodbc.connect(
            driver="{ODBC Driver 17 for SQL Server}",
            server=self.host,
            database=self.db,
            uid=self.user,
            pwd=self.pw,
        )

        return conn

    def create_session(self):
        """
        Session object.
        """

        Session = sessionmaker(bind=self.engine())
        session = Session()
        return session
