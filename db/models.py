from sqlalchemy.types import UserDefinedType
from sqlalchemy import DateTime, Column, Integer, func, VARCHAR, DOUBLE, text, cast
from sqlalchemy.orm import declarative_base

from db.connection import SQLServerConnection as ssc

import pandas as pd
from datetime import datetime


def query_converter(query_return):
    """
    The function converts sql query values to list[dict]
    """
    data_list = [dict(zip(x.keys(), x)) for x in query_return.all()]
    return data_list


class Geometry(UserDefinedType):
    """
    MSSQL Geometry datatype
    """

    cache_ok = True

    def get_col_spec(self):
        return "GEOMETRY"

    def bind_expression(self, bindvalue):
        expression = text(
            f"GEOMETRY::STGeomFromText(:{bindvalue.key}, 4326)"
        ).bindparams(bindvalue)

        return expression


# Base class
Base = declarative_base()


class ImagePredictions(Base):
    __tablename__ = "IMAGE_PREDICTIONS_SNOW"
    __table_args__ = {"schema": "BO_DATA"}

    OBJECTID = Column(Integer, primary_key=True, autoincrement=False)
    Shape = Column(Geometry)
    image_name = Column(VARCHAR)
    datetime_processed = Column(DateTime)
    prediction_prob = Column(DOUBLE)
    prediction_class = Column(VARCHAR)
    image_link_original = Column(VARCHAR)
    image_link_processed = Column(VARCHAR)
    inspection_object = Column(Integer)
    inspection_object_name = Column(VARCHAR)
    manager = Column(VARCHAR)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now())

    @classmethod
    def read_distinct(cls):
        results = pd.read_sql(
            f"SELECT distinct image_name from {cls.__table_args__['schema']}.{cls.__tablename__}",
            ssc().engine(),
        )
        return results

    @classmethod
    def insert_data(
        cls,
        id: int,
        image_name: str,
        datetime_processed: datetime,
        prediction_prob: float,
        image_link: str,
        Shape,
    ):
        """
        The function writes image logs into database

        Arguments
        ---------
        images: str
            list with blob images from Azure Storage
        log_date: str
            date when data has been read from Azure Storage
        """
        session = ssc().create_session()

        # Creating list with blob image names values
        data = cls(
            OBJECTID=id,
            image_name=image_name,
            datetime_processed=datetime_processed,
            prediction_prob=prediction_prob,
            image_link=image_link,
            Shape=Shape,
        )

        # Push data to database
        session.add(data)
        session.commit()
        session.close()

    @classmethod
    def select_max_id(cls):
        """
        The function returns max id value

        Output
        ------
        max_value: int
            max id value from vasa data table
        """
        session = ssc().create_session()
        data = session.query(func.max(cls.OBJECTID)).first()
        max_value = data[0]
        session.close()

        return max_value

    @classmethod
    def insert_records(cls, records: list[dict]) -> None:
        """
        Insert OSM data into database.
        """
        session = ssc().create_session()
        session.bulk_insert_mappings(cls, records)
        session.commit()
        session.close()


class InspectionPoints(Base):
    __tablename__ = "SNOW_INSPECTION_POINTS"
    __table_args__ = {"schema": "BO_DATA"}

    OBJECTID = Column(Integer, primary_key=True, autoincrement=False)
    Shape = Column(Geometry)
    object = Column(Integer)
    object_name = Column(VARCHAR)
    manager = Column(VARCHAR)

    @classmethod
    def get_all(cls):
        session = ssc().create_session()

        data = session.query(
            cls.object,
            cls.object_name,
            cls.manager,
            cast(cls.Shape, VARCHAR("MAX")).label("Shape"),
        )

        session.close()

        # data = query_converter(query_return=query)

        columns = data.column_descriptions
        cols = []
        for a in columns:
            cols.append(a["name"])

        data = data.all()

        list_of_dicts = []
        for x in data:
            elements = dict(zip(cols, x))
            list_of_dicts.append(elements)

        return list_of_dicts
