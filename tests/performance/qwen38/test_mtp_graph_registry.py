from .mtp_graph_registry import Alias


def test_merged_graph_keeps_later_step_events_under_outer_capture_key():
    events={4:[],12:[]};workspace={4:'smaller',12:'merged'}
    Alias(events,12)[4].append('second-step-event')
    assert events[12]==['second-step-event'] and events[4]==[]
    assert Alias(workspace,12).get(4)=='merged'
