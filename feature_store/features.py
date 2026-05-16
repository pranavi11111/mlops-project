from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

patient = Entity(
    name="patient_id",
    description="Unique patient identifier"
)

patient_source = FileSource(
    path="data/patient_features.parquet",
    timestamp_field="event_timestamp"
)

patient_features = FeatureView(
    name="patient_features",
    entities=[patient],
    ttl=timedelta(days=365),
    schema=[
        Field(name="mean_radius", dtype=Float32),
        Field(name="mean_texture", dtype=Float32),
        Field(name="mean_perimeter", dtype=Float32),
        Field(name="mean_area", dtype=Float32),
        Field(name="mean_smoothness", dtype=Float32),
        Field(name="mean_concavity", dtype=Float32),
    ],
    source=patient_source,
)