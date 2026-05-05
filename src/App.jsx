import React, { useState, useEffect, useRef } from 'react';
import { Scene } from './components/Scene';
import { rungeKutta4 } from './models/runge-kutta';
import { systemODE } from './models/dynamic-system';
import { Sequential } from './service/nn/Sequential';
import { DenseLayer } from './service/nn/DenseLayer';

// ─── Статистика и нормализация ────────────────────────────────────────────────

function computeStats(data) {
    const n = data.length;
    const means = [0, 0, 0];
    const stds  = [0, 0, 0];
    for (const row of data) {
        means[0] += row[1]; means[1] += row[2]; means[2] += row[3];
    }
    means[0] /= n; means[1] /= n; means[2] /= n;
    for (const row of data) {
        stds[0] += (row[1] - means[0]) ** 2;
        stds[1] += (row[2] - means[1]) ** 2;
        stds[2] += (row[3] - means[2]) ** 2;
    }
    stds[0] = Math.sqrt(stds[0] / n) || 1;
    stds[1] = Math.sqrt(stds[1] / n) || 1;
    stds[2] = Math.sqrt(stds[2] / n) || 1;
    return { means, stds };
}

function normalize(coords, means, stds) {
    return coords.map((v, i) => (v - means[i]) / stds[i]);
}

function denormalize(normCoords, means, stds) {
    return normCoords.map((v, i) => v * stds[i] + means[i]);
}

// ─── Модель ───────────────────────────────────────────────────────────────────

function createModel() {
    return new Sequential([
        new DenseLayer(64, 'tanh'),
        new DenseLayer(64, 'tanh'),
        new DenseLayer(3,  'linear')
    ]);
}

// ─── Нормализация производных ─────────────────────────────────────────────────
// Производная dx/dt имеет свой масштаб — нормируем отдельно через stds/h,
// чтобы target был порядка O(1), а не O(0.001).
function normalizeDeriv(deriv, stds, h) {
    // deriv = [dx, dy, dz] в исходном пространстве
    // нормируем так, чтобы величины были сопоставимы с нормализованными координатами
    return deriv.map((v, i) => v * h / stds[i]);
}

function denormalizeDeriv(normDeriv, stds, h) {
    return normDeriv.map((v, i) => v * stds[i] / h);
}

export default function App() {
    const [showReference, setShowReference] = useState(true);
    const [solution,      setSolution]      = useState([]);
    const [predictedPath, setPredictedPath] = useState([]);
    const [isTraining,    setIsTraining]    = useState(false);
    const [loss,          setLoss]          = useState(null);
    const [epochs,        setEpochs]        = useState(300);
    const [status,        setStatus]        = useState('ready');

    const modelRef   = useRef(createModel());
    const statsRef   = useRef(null);

    const params = { h: 0.01, steps: 3000 };

    const runSimulation = () => {
        const result = rungeKutta4(systemODE, 0, [0.1, 0.1, 0.1], params.h, params.steps);
        setSolution(result);
        setPredictedPath([]);
        setLoss(null);
        setStatus('ready');
    };

    useEffect(() => { runSimulation(); }, []);

    const trainModel = async () => {
        if (solution.length < 2) return;

        modelRef.current = createModel();
        setIsTraining(true);
        setPredictedPath([]);

        const { means, stds } = computeStats(solution);
        statsRef.current = { means, stds };

        const trainSize    = Math.floor(solution.length * 0.8);
        const learningRate = 0.001;
        const BATCH_SIZE   = 64;

        // ── Формируем обучающие пары: input=[x,y,z] → target=[dx/dt, dy/dt, dz/dt] ──
        // Производные берём аналитически из systemODE — это эталон.
        // Нормализуем вход (координаты) и выход (производные).
        const trainPairs = [];
        for (let i = 0; i < trainSize; i++) {
            const [t, x, y, z] = solution[i];
            const normInput  = normalize([x, y, z], means, stds);
            const rawDeriv   = systemODE(t, [x, y, z]);              // [dx, dy, dz]
            const normTarget = normalizeDeriv(rawDeriv, stds, params.h);
            trainPairs.push({ input: [normInput], target: [normTarget] });
        }

        const model = modelRef.current;

        for (let e = 0; e < epochs; e++) {
            let totalError = 0;
            const shuffled = [...trainPairs].sort(() => Math.random() - 0.5);

            // Mini-batch
            for (let b = 0; b < shuffled.length; b += BATCH_SIZE) {
                const batch = shuffled.slice(b, b + BATCH_SIZE);
                for (const { input, target } of batch) {
                    totalError += model.trainStep(input, target, learningRate);
                }
            }

            if (e % 10 === 0) {
                setLoss(totalError / trainPairs.length);
                setStatus(`epoch ${e + 1}/${epochs}`);
                await new Promise(r => setTimeout(r, 1));
            }
        }

        generatePrediction(means, stds);
        setIsTraining(false);
        setStatus('ready');
    };

    const generatePrediction = (means, stds) => {
        const model = modelRef.current;
        const h     = params.h;

        // ── Neural ODE: используем RK4, но правые части считает нейросеть ──
        // f_nn(t, [x, y, z]) → [dx/dt, dy/dt, dz/dt]
        const f_nn = (t, state) => {
            const normInput  = normalize(state, means, stds);
            const normDeriv  = model.predict([normInput])[0];          // нормализованные производные
            return denormalizeDeriv(normDeriv, stds, h);               // возвращаем в исходный масштаб
        };

        // Используем существующий rungeKutta4, подменив функцию правых частей
        const path = rungeKutta4(f_nn, 0, [0.1, 0.1, 0.1], h, params.steps);

        console.log(`Neural ODE prediction: ${path.length} points`);
        setPredictedPath(path);
    };

    return (
        <div style={{ padding: '20px', fontFamily: 'sans-serif', color: '#333' }}>
            <div style={{ marginBottom: '20px', display: 'flex', gap: '10px', alignItems: 'center' }}>
                <button
                    onClick={runSimulation}
                    style={{ padding: '10px 20px', cursor: 'pointer', background: '#007bff', color: 'white', border: 'none', borderRadius: '4px' }}
                >
                    resolve rk4
                </button>
                <button
                    onClick={() => setShowReference(p => !p)}
                    style={{
                        padding: '10px 20px', cursor: 'pointer',
                        background: showReference ? '#6c757d' : '#17a2b8',
                        color: 'white', border: 'none', borderRadius: '4px'
                    }}
                >
                    {showReference ? 'hide reference' : 'show reference'}
                </button>
                <span>STEPS: <strong>{params.steps}</strong></span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '20px' }}>
                <div style={{ border: '1px solid #ccc', borderRadius: '8px', overflow: 'hidden' }}>
                    <Scene
                        solution={showReference ? solution : []}
                        prediction={predictedPath}
                    />
                </div>

                <div style={{ background: '#f8f9fa', padding: '20px', borderRadius: '8px', border: '1px solid #dee2e6' }}>
                    <div style={{ fontSize: '14px', color: '#666' }}>
                        <p>mode: <strong>Neural ODE (RK4 + NN derivatives)</strong></p>
                        <p>status: {status}</p>
                        {loss !== null && <p>MSE: <strong>{loss.toFixed(8)}</strong></p>}
                    </div>

                    <div style={{ marginTop: '10px', fontSize: '13px' }}>
                        <label>
                            Epochs: <strong>{epochs}</strong>
                            <br />
                            <input
                                type="range" min={50} max={500} step={50}
                                value={epochs}
                                onChange={e => setEpochs(Number(e.target.value))}
                                style={{ width: '100%', marginTop: '4px' }}
                                disabled={isTraining}
                            />
                        </label>
                    </div>

                    <button
                        onClick={trainModel}
                        disabled={isTraining || solution.length === 0}
                        style={{
                            width: '100%', padding: '12px', marginTop: '10px',
                            background: isTraining ? '#ccc' : '#28a745',
                            color: 'white', border: 'none', borderRadius: '4px',
                            cursor: isTraining ? 'not-allowed' : 'pointer'
                        }}
                    >
                        {isTraining ? status : 'Start'}
                    </button>
                </div>
            </div>
        </div>
    );
}