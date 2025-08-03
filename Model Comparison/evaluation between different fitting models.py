import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, WeibullFitter, LogLogisticFitter
from lifelines.utils import concordance_index
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
from lifelines.plotting import plot_lifetimes

def calculate_ibs(model, durations, events, times):
    """计算Integrated Brier Score (IBS)，适配KaplanMeierFitter/参数模型"""
    durations = np.array(durations)
    events = np.array(events)
    if isinstance(model, KaplanMeierFitter):
        sf = interp1d(model.survival_function_.index, 
                      model.survival_function_['KM_estimate'],
                      bounds_error=False,
                      fill_value=(1.0, 0.0))
        surv_probs = np.array([sf(t) for t in times])
        preds = np.tile(surv_probs, (len(durations), 1)).T
    else:
        surv_probs = model.survival_function_at_times(times).values.flatten()
        preds = np.tile(surv_probs, (len(durations), 1)).T
    brier_scores = []
    for i, t in enumerate(times):
        y_true = ((durations <= t) & events).astype(float)
        y_pred = 1 - preds[i]
        brier = np.mean((y_true - y_pred)**2)
        brier_scores.append(brier)
    return np.trapz(brier_scores, times) / (max(times) - min(times))

def evaluate_models(df, time_threshold, n_splits=5):
    """综合评估模型性能，返回分状态的平均指标"""
    times = np.linspace(0, time_threshold, 100)
    results = {
        'Model': [], 'State': [],
        'IBS': [], 'C-index': [],
        'LogLik': [], 'AIC': [], 'BIC': []
    }
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        for state in [0, 1]:
            train_data = train[(train['State'] == state) & (train['Lasting Time'] <= time_threshold)]
            test_data = test[(test['State'] == state) & (test['Lasting Time'] <= time_threshold)]
            if len(train_data) < 10 or len(test_data) < 5:
                continue

            # 1. Kaplan-Meier
            kmf = KaplanMeierFitter().fit(train_data['Lasting Time'], event_observed=[1]*len(train_data))
            ibs = calculate_ibs(kmf, test_data['Lasting Time'], [1]*len(test_data), times)
            sf = interp1d(kmf.survival_function_.index, kmf.survival_function_['KM_estimate'],
                          bounds_error=False, fill_value=(1.0, 0.0))
            km_pred = -sf(test_data['Lasting Time'])
            cindex = concordance_index(test_data['Lasting Time'], km_pred, [1]*len(test_data))
            results['Model'].append('Kaplan-Meier')
            results['State'].append(state)
            results['IBS'].append(ibs)
            results['C-index'].append(cindex)
            results['LogLik'].append(np.nan)
            results['AIC'].append(np.nan)
            results['BIC'].append(np.nan)

            # 2. Weibull
            wf = WeibullFitter().fit(train_data['Lasting Time'], event_observed=[1]*len(train_data))
            ibs = calculate_ibs(wf, test_data['Lasting Time'], [1]*len(test_data), times)
            wf_pred = np.full(len(test_data), wf.median_survival_time_)
            cindex = concordance_index(test_data['Lasting Time'], wf_pred, [1]*len(test_data))
            results['Model'].append('Weibull')
            results['State'].append(state)
            results['IBS'].append(ibs)
            results['C-index'].append(cindex)
            results['LogLik'].append(wf.log_likelihood_)
            results['AIC'].append(wf.AIC_)
            results['BIC'].append(wf.BIC_)

            # 3. Log-Logistic
            llf = LogLogisticFitter().fit(train_data['Lasting Time'], event_observed=[1]*len(train_data))
            ibs = calculate_ibs(llf, test_data['Lasting Time'], [1]*len(test_data), times)
            llf_pred = np.full(len(test_data), llf.median_survival_time_)
            cindex = concordance_index(test_data['Lasting Time'], llf_pred, [1]*len(test_data))
            results['Model'].append('Log-Logistic')
            results['State'].append(state)
            results['IBS'].append(ibs)
            results['C-index'].append(cindex)
            results['LogLik'].append(llf.log_likelihood_)
            results['AIC'].append(llf.AIC_)
            results['BIC'].append(llf.BIC_)
    df_results = pd.DataFrame(results)
    avg_results = df_results.groupby(['Model', 'State'], as_index=False).mean(numeric_only=True)
    return avg_results

def plot_survival_curves(df, time_threshold):
    """绘制对比曲线"""
    plt.figure(figsize=(12, 6))
    for state in [0, 1]:
        data = df[(df['State'] == state) & (df['Lasting Time'] <= time_threshold)]
        if len(data) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(data['Lasting Time'], event_observed=[1]*len(data))
        kmf.plot(label=f'Kaplan-Meier (State={state})')
        wf = WeibullFitter()
        wf.fit(data['Lasting Time'], event_observed=[1]*len(data))
        wf.plot_survival_function(label=f'Weibull (State={state})')
        llf = LogLogisticFitter()
        llf.fit(data['Lasting Time'], event_observed=[1]*len(data))
        llf.plot_survival_function(label=f'Log-Logistic (State={state})')
    plt.title('Survival Function Comparison')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = "/Users/fujunhan/Desktop/RA_traffic simulation/my_research_related_materials/plan1.11/parking_data.xlsx"
    df = pd.read_excel(file_path)
    time_threshold = 75000

    results = evaluate_models(df, time_threshold)

    print("综合模型评估结果:")
    print("="*70)
    print("IBS (越小越好) | C-index (越大越好) | LogLik (越大越好) | AIC/BIC (越小越好)")
    print("="*70)
    for state in [0, 1]:
        state_results = results[results['State'] == state]
        print(f"\n状态 {state} 的模型比较:")
        print(state_results.sort_values('IBS').to_string(index=False))

    plot_survival_curves(df, time_threshold)
