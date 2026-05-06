import React, { useState, useEffect, useRef } from 'react';
import { Scene } from './components/Scene';
import { rungeKutta4 } from './models/runge-kutta';
import { systemODE } from './models/dynamic-system';
import { Sequential } from './service/nn/Sequential';
import { DenseLayer } from './service/nn/DenseLayer';

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

const normalize     = (coords, means, stds) => coords.map((v, i) => (v - means[i]) / stds[i]);
const denormalize   = (coords, means, stds) => coords.map((v, i) => v * stds[i] + means[i]);
const normDeriv     = (deriv, stds, h)      => deriv.map((v, i) => v * h / stds[i]);
const denormDeriv   = (deriv, stds, h)      => deriv.map((v, i) => v * stds[i] / h);

const createModel = () => new Sequential([
    new DenseLayer(64, 'tanh'),
    new DenseLayer(64, 'tanh'),
    new DenseLayer(3,  'linear'),
]);

export default function App() {
    const [showReference, setShowReference] = useState(true);
    const [solution,      setSolution]      = useState([]);
    const [predictedPath, setPredictedPath] = useState([]);
    const [isTraining,    setIsTraining]    = useState(false);
    const [loss,          setLoss]          = useState(null);
    const [epochs,        setEpochs]        = useState(300);
    const [status,        setStatus]        = useState('ready');

    const modelRef = useRef(createModel());
    const params   = { h: 0.01, steps: 3000 };

    const runSimulation = () => {
        setSolution(rungeKutta4(systemODE, 0, [0.1, 0.1, 0.1], params.h, params.steps));
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
        const trainSize = Math.floor(solution.length * 0.8);
        const lr        = 0.001;
        const BATCH     = 64;

        const pairs = [];
        for (let i = 0; i < trainSize; i++) {
            const [t, x, y, z] = solution[i];
            pairs.push({
                input:  [normalize([x, y, z], means, stds)],
                target: [normDeriv(systemODE(t, [x, y, z]), stds, params.h)],
            });
        }

        const model = modelRef.current;
        for (let e = 0; e < epochs; e++) {
            let err = 0;
            const shuffled = [...pairs].sort(() => Math.random() - 0.5);
            for (let b = 0; b < shuffled.length; b += BATCH) {
                for (const { input, target } of shuffled.slice(b, b + BATCH)) {
                    err += model.trainStep(input, target, lr);
                }
            }
            if (e % 10 === 0) {
                setLoss(err / pairs.length);
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
        const f_nn  = (t, state) =>
            denormDeriv(model.predict([normalize(state, means, stds)])[0], stds, h);

        setPredictedPath(rungeKutta4(f_nn, 0, [0.1, 0.1, 0.1], h, params.steps));
    };

    return (
        <div style={{ padding: '20px', fontFamily: 'sans-serif', color: '#333' }}>
            <div style={{ marginBottom: '20px', display: 'flex', gap: '10px', alignItems: 'center' }}>
                <button onClick={runSimulation}
                    style={{ padding: '10px 20px', cursor: 'pointer', background: '#007bff', color: 'white', border: 'none', borderRadius: '4px' }}>
                    resolve rk4
                </button>
                <button onClick={() => setShowReference(p => !p)}
                    style={{ padding: '10px 20px', cursor: 'pointer', background: showReference ? '#6c757d' : '#17a2b8', color: 'white', border: 'none', borderRadius: '4px' }}>
                    {showReference ? 'hide reference' : 'show reference'}
                </button>
                <span>STEPS: <strong>{params.steps}</strong></span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '20px' }}>
                <div style={{ border: '1px solid #ccc', borderRadius: '8px', overflow: 'hidden' }}>
                    <Scene solution={showReference ? solution : []} prediction={predictedPath} />
                </div>

                <div style={{ background: '#f8f9fa', padding: '20px', borderRadius: '8px', border: '1px solid #dee2e6' }}>
                    <div style={{ fontSize: '14px', color: '#666' }}>
                        <p>status: {status}</p>
                        {loss !== null && <p>MSE: <strong>{loss.toFixed(8)}</strong></p>}
                    </div>
                    <div style={{ marginTop: '10px', fontSize: '13px' }}>
                        <label>
                            Epochs: <strong>{epochs}</strong><br />
                            <input type="range" min={50} max={500} step={50} value={epochs}
                                onChange={e => setEpochs(Number(e.target.value))}
                                style={{ width: '100%', marginTop: '4px' }} disabled={isTraining} />
                        </label>
                    </div>
                    <button onClick={trainModel} disabled={isTraining || solution.length === 0}
                        style={{ width: '100%', padding: '12px', marginTop: '10px', background: isTraining ? '#ccc' : '#28a745', color: 'white', border: 'none', borderRadius: '4px', cursor: isTraining ? 'not-allowed' : 'pointer' }}>
                        {isTraining ? status : 'Start'}
                    </button>
                </div>
            </div>
        </div>
    );
}