import React, { useEffect, useMemo } from 'react';
import Chart from './chart';

import { Layout, Model, TabNode, IJsonModel } from "flexlayout-react";
import "flexlayout-react/style/light.css";
import "@blueprintjs/core/lib/css/blueprint.css";

import { AppContext, AppState, Predictor } from 'src/logic/app-state';
import Settings from './settings';

const layoutJson: IJsonModel = {
    global: {
        splitterSize: 4,
        tabEnableClose: false,
        tabSetEnableDrop: true,
        tabSetEnableDrag: true,
        tabSetEnableMaximize: true,
    },
    borders: [],
    layout: {
        type: "row",
        weight: 100,
        children: [
            {
                type: "tabset",
                weight: 60,
                children: [
                    {
                        type: "tab",
                        name: "Chart",
                        component: "chart",
                        id: "chart"
                    }
                ]
            },
            {
                type: "row",
                weight: 40,
                children: [
                    {
                        type: "tabset",
                        weight: 50,
                        children: [
                            {
                                type: "tab",
                                name: "Settings",
                                component: "settings",
                                id: "settings"
                            }
                        ]
                    },
                    // {
                    //     type: "tabset",
                    //     weight: 50,
                    //     children: [
                    //         {
                    //             type: "tab",
                    //             name: "Predictions",
                    //             component: "predictions",
                    //             id: "predictions"
                    //         }
                    //     ]
                    // }
                ]
            }
        ]
    }
};

const Predictions: React.FC = () => {
    return <div>TODO</div>
}

const App: React.FC = () => {
    const app = useMemo(() => new AppState(), [])
    const layoutModel = useMemo(() => Model.fromJson(layoutJson), []);

    const factory = (node: TabNode) => {
        const component = node.getComponent();

        switch (component) {
            case 'chart':
                return <Chart />
            case 'settings':
                return <Settings />
            case 'predictions':
                return <Predictions />
            default:
                return <div>Unknown component: {component}</div>;
        }
    };

    useEffect(() => {
        let savedPredictor: Predictor | null = null

        const onKeydown = (e: KeyboardEvent) => {
            if(e.repeat) return
            if(e.target !== document.body) return
            if(e.code === "Space") {
                savedPredictor = app.activePredictor.get()
                app.activePredictor.set({
                    id: "real"
                })
            }
        }

        const onKeyup = (e: KeyboardEvent) => {
            if(e.target !== document.body) return
            if(e.code === "Space") {
                if(!savedPredictor) return
                app.activePredictor.set(savedPredictor)
            }
        }

        document.body.addEventListener("keydown", onKeydown)
        document.body.addEventListener("keyup", onKeyup);
        return () => {
            document.body.removeEventListener("keydown", onKeydown);
            document.body.removeEventListener("keyup", onKeyup);
        };
    }, [])

    return (
        <div style={{ width: '100%', height: '100%' }}>
            <AppContext.Provider value={app}>
                <Layout
                    model={layoutModel}
                    factory={factory}
                />
            </AppContext.Provider>
        </div>
    );
}

export default App;
