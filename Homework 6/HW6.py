
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import logsumexp

FOLDER = Path(__file__).resolve().parent
TARGET_WEEK = None  

# --------------------------- PROBLEM 1(c) ---------------------------
def transition_matrix(m):
    P = np.zeros((m, m))
    P[np.arange(m - 1), np.arange(1, m)] = 1.0
    P[m - 1, :2] = 0.5
    return P


def problem_1c():
    print('\nPROBLEM 1(c)')
    print(' m     M    bound   |lambda2|')
    for m in range(3, 13):
        P = transition_matrix(m)
        pattern = P > 0
        B = pattern.copy()
        exponent = 1
        while not np.all(B):
            B = (B.astype(np.int64) @ pattern.astype(np.int64)) > 0
            exponent += 1
        moduli = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
        bound = (m - 1) ** 2 + 1
        print(f'{m:2d} {exponent:5d} {bound:8d} {moduli[1]:11.6f}')


# --------------------------- PROBLEM 2(c) ---------------------------
def game_value(score, opponent_score):
    x = (score + 1.0) / (score + opponent_score + 2.0)
    return 0.5 + 0.5 * np.sign(x - 0.5) * np.sqrt(abs(2*x - 1))


def load_games(path):
    df = pd.read_csv(path)
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace('_', ' ', regex=False)
                  .str.replace('-', ' ', regex=False))
    aliases = {
        'home pts': ['home pts', 'home points', 'home score'],
        'away pts': ['away pts', 'away points', 'away score'],
    }
    for canonical, options in aliases.items():
        matches = [c for c in options if c in df.columns]
        if matches:
            df = df.rename(columns={matches[0]: canonical})
    needed = ['season', 'week', 'home', 'home pts', 'away', 'away pts', 'completed']
    missing = [x for x in needed if x not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}. Found: {df.columns.tolist()}')
    df['season'] = pd.to_numeric(df['season'], errors='raise').astype(int)
    df['week'] = pd.to_numeric(df['week'], errors='raise').astype(int)
    df['completed'] = df['completed'].astype(str).str.strip().str.lower().isin(['1', 'true', 'yes'])
    df['home'] = df['home'].astype(str).str.strip()
    df['away'] = df['away'].astype(str).str.strip()
    df['home pts'] = pd.to_numeric(df['home pts'], errors='coerce')
    df['away pts'] = pd.to_numeric(df['away pts'], errors='coerce')
    return df


def priors_from_2025(games, teams):
    m = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    prior_flat = np.full((m, m), 0.5)
    last_sum = np.zeros((m, m))
    last_count = np.zeros((m, m))
    wins = np.zeros(m)
    played = np.zeros(m)
    for _, g in games.iterrows():
        if g['home'] not in idx or g['away'] not in idx:
            continue
        i, j = idx[g['home']], idx[g['away']]
        hi = game_value(g['home pts'], g['away pts'])
        last_sum[i,j] += hi
        last_sum[j,i] += 1-hi
        last_count[i,j] += 1
        last_count[j,i] += 1
        played[[i,j]] += 1
        if g['home pts'] > g['away pts']:
            wins[i] += 1
        elif g['away pts'] > g['home pts']:
            wins[j] += 1
        else:
            wins[[i,j]] += 0.5
    last = np.divide(last_sum, last_count, out=prior_flat.copy(), where=last_count>0)
    s = (wins + 1) / (played + 2)
    custom = s[:, None] / (s[:, None] + s[None, :])
    for prior in (prior_flat, last, custom):
        np.fill_diagonal(prior, 0)
    return {'Flat': prior_flat, '2025 meetings': last, 'Custom 2025 win rates': custom}


def keener(games, teams, prior, w=1.0):
    m = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    A = w * prior.copy()
    count = np.zeros(m)
    for _, g in games.iterrows():
        i, j = idx[g['home']], idx[g['away']]
        h = game_value(g['home pts'], g['away pts'])
        A[i,j] += h
        A[j,i] += 1-h
        count[[i,j]] += 1
    A /= (count + w*(m-1))[:, None]
    np.fill_diagonal(A, 0)
    eigenvalues = np.linalg.eigvals(A)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    leading = np.real(eigenvalues[order[0]])
    ratio = abs(eigenvalues[order[1]]/leading)
    r = np.ones(m) / m
    for _ in range(100000):
        new_r = A @ r
        new_r /= new_r.sum()
        if np.max(np.abs(new_r-r)) < 1e-13:
            r = new_r
            break
        r = new_r
    return leading, ratio, r


def problem_2c():
    filepath = FOLDER / 'big12.csv'
    print('\nPROBLEM 2(c)')
    if not filepath.exists():
        print('Skipped: place Canvas big12.csv beside this script, then rerun.')
        return
    df = load_games(filepath)
    seasons = df[df.season.isin([2025,2026])]
    current = seasons[seasons.season == 2026]
    teams = sorted(set(current.home).union(current.away))
    if len(teams) != 16:
        print(f'Warning: found {len(teams)} 2026 teams, expected 16.')
    hist2025 = seasons[(seasons.season == 2025) & seasons.completed].dropna(subset=['home pts','away pts'])
    completed2026 = current[current.completed].dropna(subset=['home pts','away pts'])
    priors = priors_from_2025(hist2025, teams)
    for name, prior in priors.items():
        eig, ratio, r = keener(completed2026, teams, prior)
        print(f'\n{name}: lambda*={eig:.8f}; |lambda2/lambda*|={ratio:.8f}')
        for t, score in sorted(zip(teams,r), key=lambda x:-x[1]):
            print(f'  {t:25s} {score:.6f}')
        plt.figure(figsize=(12, 6))
        order = np.argsort(r)[::-1]
        plt.bar(np.array(teams)[order], r[order])
        plt.xticks(rotation=65, ha='right')
        plt.ylabel('Perron ranking score (sums to 1)')
        plt.title(f'2026 Big 12: {name}')
        plt.tight_layout()
        filename = 'q2c_' + name.lower().replace(' ', '_') + '.png'
        plt.savefig(FOLDER / filename, dpi=180)
        plt.close()
        print('Saved:', filename)

    print('\n2026 week numbers:', sorted(current.week.unique().tolist()))
    if TARGET_WEEK is None:
        print('To predict October 10-11 games, set TARGET_WEEK at top of script.')
        print('Set it to the week number of those games in Canvas big12.csv.')
        return
    scheduled = current[current.week == TARGET_WEEK]
    earlier = completed2026[completed2026.week < TARGET_WEEK]
    if scheduled.empty:
        print(f'No games found for week {TARGET_WEEK}. Check TARGET_WEEK.')
        return
    for name, prior in priors.items():
        _, _, r = keener(earlier, teams, prior)
        scores = dict(zip(teams,r))
        print(f'\nPre-week {TARGET_WEEK} predictions: {name}')
        correct = total = 0
        for _, g in scheduled.iterrows():
            predicted = g['home'] if scores[g['home']] >= scores[g['away']] else g['away']
            if g['completed'] and pd.notna(g['home pts']) and pd.notna(g['away pts']):
                actual = (g['home'] if g['home pts'] > g['away pts'] else
                          g['away'] if g['away pts'] > g['home pts'] else 'TIE')
                if actual != 'TIE':
                    total += 1
                    correct += predicted == actual
            else:
                actual = 'Not played / not updated'
            print(f"  {g['away']} at {g['home']}: prediction={predicted}, result={actual}")
        if total:
            print(f'Accuracy: {correct}/{total} ({correct/total:.1%})')
        else:
            print('No completed games recorded yet for this week.')


# --------------------------- PROBLEM 3(c) ---------------------------
def outbreak_probability(N, R0):
    K = N//10
    k = np.arange(1,K)
    log_rho = np.r_[0.0, np.cumsum(np.log(N / (R0*(N-k))))]
    return float(np.exp(-logsumexp(log_rho)))


def problem_3c():
    grid = np.linspace(0.5,3,400)
    plt.figure(figsize=(9,5))
    for N in [100,1000]:
        plt.plot(grid, [outbreak_probability(N,R) for R in grid],label=f'Exact N={N}')
    # Assignment explicitly asks for 1 - 1/R0, including its negative part.
    plt.plot(grid, 1-1/grid, 'k--', label='1 - 1/R0')
    plt.xlabel('R0')
    plt.ylabel('h1 (probability)')
    plt.title('Problem 3(c): Reaching outbreak threshold')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FOLDER/'q3c.png',dpi=180)
    plt.close()
    print('\nPROBLEM 3(c): saved q3c.png')


# --------------------------- PROBLEM 3(d) ---------------------------
def expected_physical_time(R0, N=100, K=10, gamma=1, dt=0.005):
    n = np.arange(1,K)
    beta = R0*gamma/N
    p = beta*n*(N-n)*dt
    q = gamma*n*dt
    A = np.diag(p+q)+np.diag(-p[:-1],1)+np.diag(-q[1:],-1)
    e = np.linalg.solve(A,np.ones(K-1))
    return float(e[0]*dt), float(e[0])


def simulate_time(R0, trials=20000, N=100, K=10, dt=0.005, seed=42):
    rng = np.random.default_rng(seed)
    states = np.ones(trials,dtype=np.int32)
    steps = np.zeros(trials,dtype=np.int32)
    while True:
        active = np.flatnonzero((states>0)&(states<K))
        if not len(active):
            break
        n=states[active]
        up = (R0/N)*n*(N-n)*dt
        down=n*dt
        u=rng.random(len(active))
        states[active] += (u<up).astype(np.int32)-(u>=1-down).astype(np.int32)
        steps[active] += 1
    return float(steps.mean()*dt)


def problem_3d():
    print('\nPROBLEM 3(d): R0, e1 (steps), e1*dt (time)')
    for R0 in [0.5,0.8,1.5,3.0]:
        physical, steps = expected_physical_time(R0)
        print(f' {R0:3.1f}   {steps:10.4f}   {physical:10.6f}')
    simulated = simulate_time(1.5)
    exact, _ = expected_physical_time(1.5)
    print(f'Simulation check R0=1.5: simulated={simulated:.6f}; exact={exact:.6f}')
    print('e1*dt independent of dt because p and q are each proportional to dt.')


if __name__ == '__main__':
    problem_1c()
    problem_2c()
    problem_3c()
    problem_3d()
