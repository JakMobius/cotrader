import React from 'react';
import { Predictor } from '../logic/app-state';

interface PredictorItemProps {
    predictor: Predictor;
    isActive: boolean;
    onSelect: (predictor: Predictor) => void;
}

export const PredictorItem: React.FC<PredictorItemProps> = ({ predictor, isActive, onSelect }) => {
    return (
        <div
            className={`predictor-item ${isActive ? 'active' : ''}`}
            onClick={() => onSelect(predictor)}
        >
            <h4>{predictor.name ?? predictor.id}</h4>
        </div>
    );
};

export default PredictorItem;
