import React, { useEffect, useState } from 'react';
import { useApp } from '../logic/app-state';
import PredictorSelector from './predictor-selector';
import './settings.css';

// Helper functions to format date and time for input fields
const formatDateForInput = (date: Date): string => {
    return date.toISOString().split('T')[0];
};

const formatTimeForInput = (date: Date): string => {
    return date.toTimeString().substring(0, 5); // HH:MM format in 24h notation
};

const DateTimePicker: React.FC = () => {
    const app = useApp();
    const dateLimit = app.dateLimit.use();
    const onlineMode = app.onlineMode.use();
    const snapChart = app.snapChart.use();

    const [inputDate, setInputDate] = useState(formatDateForInput(dateLimit));
    const [inputTime, setInputTime] = useState(formatTimeForInput(dateLimit));

    useEffect(() => {
        setInputDate(formatDateForInput(dateLimit));
        setInputTime(formatTimeForInput(dateLimit));
    }, [dateLimit]);

    const handleDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newDate = e.target.value;
        if (newDate) {
            const updatedDate = new Date(dateLimit);
            const [year, month, day] = newDate.split('-').map(Number);
            updatedDate.setFullYear(year, month - 1, day); // month is 0-indexed in JS Date
            app.dateLimit.set(updatedDate);
        }
    };

    const handleTimeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newTime = e.target.value;
        if (newTime) {
            const updatedDate = new Date(dateLimit);
            const [hours, minutes] = newTime.split(':').map(Number);
            updatedDate.setHours(hours, minutes);
            app.dateLimit.set(updatedDate);
        }
    };

    return (
        <section className="section">
            <div className="datetime-header">
                <label className="follow-limit-checkbox">
                    <input
                        type="checkbox"
                        checked={!onlineMode}
                        onChange={(e) => app.onlineMode.set(!e.target.checked)}
                    />
                    <span>Cutoff Date</span>
                </label>
            </div>
            {!onlineMode && (
                <div className="datetime-container">
                    <input
                        type="date"
                        value={inputDate}
                        onChange={handleDateChange}
                        className="date-input"
                    />
                    <input
                        type="time"
                        value={inputTime}
                        onChange={handleTimeChange}
                        className="time-input"
                    />
                </div>
            )}
            {onlineMode && (
                <div className="snap-chart-control">
                    <label className="snap-chart-checkbox">
                        <input
                            type="checkbox"
                            checked={snapChart}
                            onChange={(e) => app.snapChart.set(e.target.checked)}
                        />
                        <span>Follow Latest Candle</span>
                    </label>
                </div>
            )}
        </section>
    );
};

const intervals = [
    '1s', '5s', '10s', '30s',
    '1m', '3m', '5m', '15m', '30m',
    '1h', '2h', '4h', '6h', '8h', '12h',
    '1d', '3d', '1w', '1M'
];

const TradingControls: React.FC = () => {
    const app = useApp();
    const symbol = app.symbol.use();
    const interval = app.interval.use();
    const [inputSymbol, setInputSymbol] = useState(symbol);

    useEffect(() => {
        setInputSymbol(symbol);
    }, [symbol]);

    const handleSymbolChange = (newSymbol: string) => {
        if (newSymbol.trim()) {
            app.symbol.set(newSymbol.toUpperCase());
        }
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setInputSymbol(e.target.value.toUpperCase());
    };

    const handleInputBlur = () => {
        handleSymbolChange(inputSymbol);
    };

    const handleSymbolSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        handleSymbolChange(inputSymbol);
    };

    const handleIntervalChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        app.interval.set(e.target.value);
    };

    return (
        <section className="section">
            <div className="trading-controls">
                <div className="input-group">
                    <h3>Trading Pair</h3>
                    <form onSubmit={handleSymbolSubmit} className="custom-symbol-form">
                        <input
                            type="text"
                            value={inputSymbol}
                            onChange={handleInputChange}
                            onBlur={handleInputBlur}
                            onKeyDown={(e) => e.key === 'Enter' && handleSymbolSubmit(e)}
                            placeholder="Enter trading pair (e.g., BTCUSDT)"
                            className="custom-symbol-input"
                        />
                    </form>
                </div>

                <div className="input-group">
                    <h3>Interval</h3>
                    <select
                        value={interval}
                        onChange={handleIntervalChange}
                        className="interval-select"
                    >
                        {intervals.map(int => (
                            <option key={int} value={int}>
                                {int}
                            </option>
                        ))}
                    </select>
                </div>
            </div>
        </section>
    );
};

export const Settings: React.FC = () => {
    return (
        <div className="settings-container">
            <TradingControls />
            <DateTimePicker />
            <PredictorSelector />
        </div>
    );
};

export default Settings;
