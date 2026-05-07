% Engram KeyStore — auto-generated, do not edit by hand.
:- dynamic(key_memory/2).
:- dynamic(relationship/3).
:- dynamic(fact/4).
:- dynamic(belief/3).

% key_memory(Id, Text).
key_memory('guard_backstory_3', 'Miguel enlisted the year after Tomas died. Said he wanted a soldier\'s death rather than a sick man\'s. He got his wish. We received word in the autumn — no body, just a letter from his captain.').
key_memory('guard_backstory_0', 'My father left when I was eight. One morning he was there; by evening he was gone with no word and no reason. I learned early that men disappear.').
key_memory('guard_backstory_2', 'Tomas — my elder brother — died of the coughing sickness in the winter of \'43. He was twenty-two. I watched him shrink to nothing over three months and I could do nothing to stop it.').
key_memory('guard_backstory_1', 'I started on the docks at ten, hauling rope for a penny a day. The harbour master beat boys who were slow. I learned to be fast and invisible.').
key_memory('f593c921-df82-4b99-bd53-197814c44d91', 'Player revealed Mayor Fletcher has been embezzling funds, warning me to be careful.').

% fact(NpcId, Subject, Predicate, Object).
relationship(rico, sofia, ally).
fact(rico, tomas, status, deceased).
fact(rico, miguel, status, deceased).
fact(rico, father, status, absent).
belief(rico, docks_are_dangerous, true).
belief(rico, strangers_want_something, true).
