% Engram KeyStore — auto-generated, do not edit by hand.
:- dynamic(key_memory/2).
:- dynamic(relationship/3).
:- dynamic(fact/4).
:- dynamic(belief/3).

% key_memory(Id, Text).
key_memory('clerk_backstory_0', 'My father left when I was eight. One morning he was there; by evening he was gone with no word and no reason. I learned early that men disappear.').
key_memory('clerk_backstory_2', 'Tomas — my elder brother — died of the coughing sickness in the winter of \'43. He was twenty-two. I watched him shrink to nothing over three months and I could do nothing to stop it.').
key_memory('clerk_backstory_5', 'The first time I moved untaxed cargo past the harbormaster I was nineteen. A bolt of French silk hidden in a barrel of dried fish. My hands shook the whole way through the gate. After that, they never shook again.').
key_memory('6fc619d3-6db2-4471-b680-d80f40144d34', 'The player requested financial assistance for a sick child, which Rico agreed to consider.').
key_memory('0b9673b8-e0dd-48a3-8d56-a7eb2f1238ce', 'Agreed to consider financial assistance for a sick child after the player detailed the child\'s worsening fever.').

% fact(NpcId, Subject, Predicate, Object).
relationship(rico, sofia, ally).
fact(rico, tomas, status, deceased).
fact(rico, miguel, status, deceased).
fact(rico, father, status, absent).
belief(rico, docks_are_dangerous, true).
belief(rico, strangers_want_something, true).
