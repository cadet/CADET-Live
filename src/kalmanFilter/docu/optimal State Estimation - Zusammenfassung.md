# Optimal State Estimation - Zusammenfassung

Dies ist eine strukturierte Zusammenfassung des Buches "Optimal State Estimation" von Dan Simon (2006).

---

# Kapitel 1: Einführung in die Zustandschätzung

## 1.1 Motivation
- Zustände können oft nicht direkt gemessen werden.
- Typische Anwendungen: Robotik, Flugnavigation, Automotive, Medizintechnik, Prozessregelung.
- Ziel: Bestmögliche Schätzung des Zustands $\hat{x}(t)$ anhand der verfügbaren Messungen.

## 1.2 Systemmodelle
- Kontinuierliches System: $\dot{x}(t) = f(x(t),u(t),t) + w(t)$
- Diskrete Messung: $y_k = h(x(t_k),t_k) + v_k$
- $w(t)$ = Prozessrauschen, $v_k$ = Messrauschen, üblicherweise Gaussian.

## 1.3 Schätzsysteme
- Zwei Schritte: Vorhersage (Predict) und Korrektur (Correct).
- Architektur:
```
      Prozess                Schätzer
x → f(·) → x + w   ---->    Prediction
                      ↑
                 y + v →    Correction
```

## 1.4 Optimalität
- Filter ist optimal, wenn er MMSE minimiert:
$J = E[e_k^\top e_k]$ mit $e_k = x_k - \hat{x}_k$
- Kalman-Filter: optimaler linearer MMSE-Schätzer.

## 1.5 Aufbau des Buches
- Kapitel 1–4: Grundlagen
- Kapitel 5–8: Kalman-Filter, EKF, UKF
- Kapitel 9–15: Erweiterte und robuste Filter
- Kapitel 16–20: Implementierung und Praxisbeispiele

---

# Kapitel 2: Mathematische Grundlagen der Schätzung

## 2.1 Wahrscheinlichkeitstheorie
- Erwartungswert $E[x]$, Varianz $Var(x)$, Kovarianz $Cov(x,y)$
- Transformationen: $Cov(Ax,By) = A Cov(x,y) B^\top$

## 2.2 Gaußverteilungen
- $x \sim \mathcal{N}(\mu, P)$
- Lineare Transformation bleibt Gaussian: $y = Ax + b \Rightarrow y \sim \mathcal{N}(A\mu+b, APA^\top)$

## 2.3 Lineare Regression / Least-Squares
- Minimierung von $J = (y - Hx)^\top R^{-1} (y - Hx)$
- Lösung: $\hat{x} = (H^\top R^{-1} H)^{-1} H^\top R^{-1} y$
- KF als rekursive LS-Methode (RLS)

## 2.4 Bayes’sche Schätzung
- Posterior: $p(x|y) = \frac{p(y|x)p(x)}{p(y)}$
- Prior = Prediction, Likelihood = Messmodell, Posterior = aktualisierte Schätzung

## 2.5 Maximum Likelihood & MAP
- ML: $\hat{x}_{ML} = \arg\max_x p(y|x)$
- MAP: $\hat{x}_{MAP} = \arg\max_x p(x|y)$
- MAP = KF für Gaussian

## 2.6 Fehlerpropagation
- Für $y = Ax + b + w$: $P_y = A P_x A^\top + Q$

---

# Kapitel 3: Dynamische Systeme

## 3.1 Zustandsraumdarstellung
- Lineares diskretes System: $x_{k+1} = F_k x_k + G_k u_k + w_k$, $y_k = H_k x_k + v_k$
- Kontinuierlich: $\dot{x} = A x + B u + w$

## 3.2 Transition Matrix
- Kontinuierliche Lösung: $x(t) = \Phi(t,t_0)x(t_0) + \int_{t_0}^t \Phi(t,\tau)Bu(\tau)d\tau$
- Für konstantes A: $\Phi(t) = e^{At}$

## 3.3 Diskretisierung
- Diskrete Systemmatrizen: $F = e^{A\Delta t}$
- Diskretes Prozessrauschen: $Q_d = \int_0^{\Delta t} e^{A\tau} Q e^{A^\top \tau} d\tau$
- Van-Loan Methode:
$\exp\begin{pmatrix}-A & Q \\ 0 & A^\top\end{pmatrix} \Delta t = \begin{pmatrix}F & Q_d \\ 0 & F^{-T}\end{pmatrix}$

## 3.4 Prozess- und Messrauschen
- Prozessrauschen modelliert ungewisse Dynamik
- Messrauschen modelliert Sensorgenauigkeit
- Dient oft als Modellierungswerkzeug

## 3.5 Stabilität
- Eigenwerte von A bzw F bestimmen Stabilität
- KF bleibt unter milden Bedingungen stabil

## 3.6 Beobachtbarkeit & Steuerbarkeit
- Observability-Matrix $\mathcal{O} = \begin{pmatrix}H \\ HF \\ HF^2 \\ ...\end{pmatrix}$
- Rang = n für volle Beobachtbarkeit

---

# Kapitel 4: Lineare diskrete Kalman-Filter

## 4.1 Prediction
$\hat{x}_{k|k-1} = F_k \hat{x}_{k-1|k-1} + G_k u_k$
$P_{k|k-1} = F_k P_{k-1|k-1} F_k^\top + Q_k$

## 4.2 Innovation & Kalman-Gain
$\tilde{y}_k = y_k - H_k \hat{x}_{k|k-1}$
$S_k = H_k P_{k|k-1} H_k^\top + R_k$
$K_k = P_{k|k-1} H_k^\top S_k^{-1}$

## 4.3 Correction
$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k \tilde{y}_k$
$P_{k|k} = (I - K_k H_k) P_{k|k-1}$

---

# Kapitel 5: Kontinuierlicher Kalman-Filter

- System: $\dot{x} = A x + B u + w(t)$, $y = H x + v$
- Kovarianz-DGL: $\dot{P} = A P + P A^T + Q$
- Filtergleichung: $\dot{\hat{x}} = A\hat{x} + Bu + K(t)(y - H\hat{x})$, $K(t) = P H^T R^{-1}$
- Diskretisierung für Implementierung notwendig

---

# Kapitel 6: Erweiterter Kalman-Filter (EKF)

## 6.1 Nichtlineare Modelle
$x_{k+1} = f(x_k,u_k) + w_k$
$y_k = h(x_k) + v_k$

## 6.2 Idee des EKF
- Linearisierung um aktuelle Schätzung:
$A_k = \partial f/\partial x|_{\hat{x}}$, $H_k = \partial h/\partial x|_{\hat{x}_{k|k-1}}$

## 6.3 EKF-Gleichungen
**Prediction:**
$\hat{x}_{k|k-1} = f(\hat{x}_{k-1|k-1},u_k)$
$P_{k|k-1} = A_{k-1} P_{k-1|k-1} A_{k-1}^\top + Q$

**Update:**
$\tilde{y}_k = y_k - h(\hat{x}_{k|k-1})$
$S_k = H_k P_{k|k-1} H_k^\top + R$
$K_k = P_{k|k-1} H_k^\top S_k^{-1}$
$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k \tilde{y}_k$
$P_{k|k} = (I - K_k H_k) P_{k|k-1}$

## 6.4 Stabilität
- EKF nicht global stabil, Fehler in Jacobian propagieren direkt
- Ursachen für Divergenz: zu starke Nichtlinearität, kleine Q/R, schlechte Initialisierung

## 6.5 Varianten
- Iterated EKF: mehrfache Korrekturschritte
- Second-Order EKF: nutzt 2. Ableitungen

---

# Kapitel 7: Unscented Kalman-Filter (UKF)

- Propagiert Sigma-Punkte statt linearisieren
- Sigma-Punkte: $x^{(i)} = \hat{x} \pm \sqrt{(n+\lambda)P}$
- Transformation durch nichtlineares Modell
- Mittelwert & Kovarianz aus transformierten Punkten rekonstruieren
- Vorteile: höhere Genauigkeit bei nichtlinearen Systemen, kein Jacobian nötig

---

# Kapitel 8: Sigma-Point-Filter und Vergleich EKF/UKF

- Vergleich EKF, UKF, Partikelfilter:
  - EKF: linearisiert, einfacher, instabil bei starker Nichtlinearität
  - UKF: Sigma-Punkte, stabiler, präziser
  - PF: Monte-Carlo, flexibel, teuer
- Sigma-Point-Filter bildet Grundlage für viele moderne Navigation/Fusion Algorithmen

---

# Zusammenfassung

| Kapitel | Inhalt | Relevanz |
|--------|--------|----------|
| 1 | Einführung, Motivation, Architektur | Motivation für Filterdesign |
| 2 | Wahrscheinlichkeitsgrundlagen, Gauß, LS, Bayes | Basis für KF/EKF/UKF |
| 3 | Dynamische Systeme, Zustandsraum, Diskretisierung | Grundlage für Filtermatrizen |
| 4 | Diskreter KF | Standardfilter für lineare Systeme |
| 5 | Kontinuierlicher KF | Filtergleichung für ODEs |
| 6 | EKF | Nichtlineare Erweiterung, lokaler Taylor-Approx. |
| 7 | UKF | Sigma-Punkte, höhere Genauigkeit |
| 8 | Vergleich, Sigma-Point-Filter | Entscheidungsgrundlage für Filterwahl |

---

# Hinweise
- Filterwahl hängt von Nichtlinearität, Rauschmodell, Rechenaufwand ab.
- Stabilität und Initialisierung entscheidend.
- Praktische Implementierung: Joseph-Form für P, Resampling beim PF, numerische Stabilität beachten.

