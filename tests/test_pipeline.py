import os
os.environ['DATA_MODE']='demo'
from pathlib import Path
from backend.app.data.demo import generate_demo_dataset
from backend.app.data.adapters import DynamicSupplyChainAdapter
from backend.app.ml.training import train_all, retrieve
from backend.app.services.insights import route_recommend

def test_end_to_end_demo(tmp_path):
    p=generate_demo_dataset(tmp_path/'dynamic_supply_chain_logistics_dataset.csv',n=180)
    df=DynamicSupplyChainAdapter(source_name='SYNTHETIC_DEMO_FIXTURE').load(p)
    assert len(df)==180 and df.traffic_level.notna().all()
    m=train_all(df)
    assert m['rows']==180
    hits=retrieve('high traffic delayed route',3)
    assert len(hits)==3 and 'score' in hits[0]

def test_route_ranking():
    r=route_recommend([{'route_id':'a','route_risk':8,'traffic_level':8,'shipping_cost_usd':100},{'route_id':'b','route_risk':2,'traffic_level':2,'shipping_cost_usd':120}])
    assert r[0]['route_id']=='b'
