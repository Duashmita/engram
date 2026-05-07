% Engram KeyStore — auto-generated, do not edit by hand.
:- dynamic(key_memory/2).
:- dynamic(relationship/3).
:- dynamic(fact/4).
:- dynamic(belief/3).

% key_memory(Id, Text).
key_memory('merchant_backstory_0', 'My father left when I was eight. One morning he was there; by evening he was gone with no word and no reason. I learned early that men disappear.').
key_memory('merchant_backstory_2', 'Tomas — my elder brother — died of the coughing sickness in the winter of \'43. He was twenty-two. I watched him shrink to nothing over three months and I could do nothing to stop it.').
key_memory('merchant_backstory_5', 'The first time I moved untaxed cargo past the harbormaster I was nineteen. A bolt of French silk hidden in a barrel of dried fish. My hands shook the whole way through the gate. After that, they never shook again.').
key_memory('merchant_backstory_1', 'I started on the docks at ten, hauling rope for a penny a day. The harbour master beat boys who were slow. I learned to be fast and invisible.').
key_memory('45c88660-63f2-40cb-806f-138ca3e7d8ab', 'Player asked for help with a sick child, Rico offered assistance.').

% fact(NpcId, Subject, Predicate, Object).
relationship(rico, sofia, ally).
fact(rico, tomas, status, deceased).
fact(rico, miguel, status, deceased).
fact(rico, father, status, absent).
belief(rico, docks_are_dangerous, true).
belief(rico, strangers_want_something, true).
