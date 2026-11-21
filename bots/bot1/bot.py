"""A calling station: never folds, never raises. Something simple to play against."""

nodes = 0;
class node:
    def __init__(self,childrenstates,probs):
        self.childrenstates = {} # dictionary
        self.probs = {} # dictionary
        self.id = nodes 
        nodes += 1 
    def addchild(child):
        self.childrenstates[]


def decide(game_state: dict) -> dict:

    cards = game_state.get("your_cards")
    


    
    if game_state.get("can_check"):
        return {"action": "check"}
    return {"action": "call"}
