import React, { useEffect, useState } from 'react';
import { useApp, Predictor } from '../logic/app-state';

interface PredictorItemProps {
    predictor: Predictor;
    onSelect: (predictor: Predictor) => void;
}

const PredictorItem: React.FC<PredictorItemProps> = ({ predictor, onSelect }) => {
    const app = useApp()
    const predictorSource = app.predictionCandleStore
    const activePredictor = app.activePredictor.use()

    const [progress, setProgress] = useState<number>(0)
    const [completed, setCompleted] = useState<boolean>(false)
    const isActive = activePredictor?.id == predictor.id

    useEffect(() => {
        setProgress(0)
        setCompleted(false)

        const onReset = () => {
            setProgress(0)
            setCompleted(predictorSource.hasDataFor(predictor.id))
        }

        const onProgress = () => {
            if (predictorSource.model !== predictor.id) return
            setProgress(predictorSource.getProgress())
        }

        const onFinished = () => {
            if (predictorSource.model !== predictor.id) return
            setCompleted(predictorSource.hasDataFor(predictor.id))
            setProgress(0)
        }

        predictorSource.on("reset", onReset)
        predictorSource.on("new-chunk", onProgress)
        predictorSource.on("finish", onFinished)

        return () => {
            predictorSource.off("reset", onReset);
            predictorSource.off("new-chunk", onProgress);
            predictorSource.off("finish", onFinished);
        };
    }, [predictor])

    return (
        <div
            className={`predictor-item ${isActive ? 'active' : ''}`}
            onClick={() => onSelect(predictor)}
        >
            <div className="progressbar" style={{
                width: progress * 100 + "%"
            }}></div>
            <div className="text">{predictor.name ?? predictor.id}</div>
            <div className={"completion-indicator" + (completed ? " visible" : "")}></div>
        </div>
    );
};

interface PredictorSelectorProps {
    className?: string;
}

export const PredictorSelector: React.FC<PredictorSelectorProps> = ({ className }) => {
    const app = useApp();
    const activePredictor = app.activePredictor.use();

    const [predictors, setPredictors] = useState<Predictor[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const fetchPredictors = async () => {
            setLoading(true);
            try {
                const result = await app.api.getPredictors();
                setPredictors(result);
            } catch (error) {
                console.error("Failed to fetch predictors:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchPredictors();
    }, [app.api]);

    const handlePredictorSelect = (predictor: Predictor) => {
        app.activePredictor.set(predictor.id === activePredictor?.id ? null : predictor);
    };

    return (
        <section className={`section ${className || ''}`}>
            <h3>Predictors</h3>
            {loading ? (
                <div className="loading">Loading predictors...</div>
            ) : (
                <div className="predictor-list">
                    {predictors.map(predictor => (
                        <PredictorItem
                            key={predictor.id}
                            predictor={predictor}
                            onSelect={handlePredictorSelect}
                        />
                    ))}
                    {predictors.length === 0 && (
                        <div className="no-predictors">No predictors available</div>
                    )}
                </div>
            )}
        </section>
    );
};

export default PredictorSelector;
