:- set_prolog_flag(verbose, silent).
% Engram built-in OCEAN memory-strength rules — auto-generated.

apply_openness_filter(Novelty, O, FN) :- FN is Novelty * (0.3 + 0.7 * O).
apply_social_weight(Social, E, WS) :- WS is Social * (0.2 + 0.8 * E).
apply_threat_sensitivity(Threat, N, TS) :-
    Raw is Threat * (0.3 + 1.2 * N),
    (Raw > 1.0 -> TS = 1.0 ; TS = Raw).
apply_goal_relevance(Goal, C, GW) :- GW is Goal * (0.2 + 0.8 * C).
apply_cooperative_bias(Social, Emotion, A, CW) :-
    (Emotion >= 0 -> CW is Social * (0.5 + 0.5 * A)
     ; CW is Social * (0.5 + 0.5 * (1.0 - A))).
compute_strength(FN, WS, TS, GW, CW, SR, S) :-
    Raw is (FN*0.15 + WS*0.15 + TS*0.25 + GW*0.2 + CW*0.1 + SR*0.15),
    (Raw > 1.0 -> S = 1.0 ; S = Raw).
process_memory(O, C, E, A, N, EV, SS, TL, GR, NL, SR, Strength) :-
    apply_openness_filter(NL, O, FN),
    apply_social_weight(SS, E, WS),
    apply_threat_sensitivity(TL, N, TS),
    apply_goal_relevance(GR, C, GW),
    apply_cooperative_bias(SS, EV, A, CW),
    compute_strength(FN, WS, TS, GW, CW, SR, Strength).
