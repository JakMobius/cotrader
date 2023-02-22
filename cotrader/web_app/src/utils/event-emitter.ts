
export type Handler = (...params: any[]) => any

// TODO: it's probably better to use linked list here.
// AbstractEventHandlerSet can also be made a lot faster
// with O(n) for target change instead of O(n^2).

export interface HandlerList {
    handlers: Array<Handler>
    iterationIndex: number
    iterationCount: number
}

export default class EventEmitter {

    private handlers: Map<string, HandlerList>[]

    public static PRIORITY_LOW = 3
    public static PRIORITY_MONITOR = 2
    public static PRIORITY_NORMAL = 1
    public static PRIORITY_HIGH = 0

    constructor() {
        this.handlers = []
    }

    addListener(type: string, listener: () => void, priority: number) {
        let priorityBlock = this.handlers[priority]

        if(!priorityBlock) {
            priorityBlock = new Map()
            this.handlers[priority] = priorityBlock
        }

        let handlers = priorityBlock.get(type)
        if(handlers) {
            handlers.handlers.push(listener)
        } else {
            priorityBlock.set(type, this.handlerList([listener]))
        }
    }

    on(type: string, listener: (...params: any[]) => any, priority: number = EventEmitter.PRIORITY_NORMAL){
        return this.addListener(type, listener, priority);
    }

    removeListener(type: string, listener: (...params: any[]) => any) {

        for(let priorityBlock of this.handlers) {
            if(!priorityBlock) continue
            let handlerList = priorityBlock.get(type)
            if(!handlerList) continue
            let index = handlerList.handlers.indexOf(listener)
            if(index !== -1) {
                handlerList.handlers.splice(index, 1)
                if(index < handlerList.iterationCount) handlerList.iterationCount--
                if(index <= handlerList.iterationIndex) handlerList.iterationIndex--
            }
        }
    }

    off(type: string, listener: (...params: any[]) => any) {
        return this.removeListener(type, listener)
    }

    removeAllListeners(): void
    removeAllListeners(type?: string): void {
        if(type) {
            for (let priorityBlock of this.handlers) {
                if (type && priorityBlock) {
                    priorityBlock.delete(type)
                }
            }
        } else {
            this.handlers = []
        }
    }

    once(type: string, listener: (...params: any[]) => any) {
        let self = this
        function on() {
            self.removeListener(type, on);
            return listener.apply(self, Array.from(arguments));
        }
        on.listener = listener;
        return this.on(type, on);
    }

    emitArgs(type: string, args?: any[]) {
        let result = true;

        for(let priorityBlock of this.handlers) {
            if(!priorityBlock) continue
            let handlerList = priorityBlock.get(type)
            if(!handlerList) continue

            handlerList.iterationIndex = 0
            handlerList.iterationCount = handlerList.handlers.length

            for(; handlerList.iterationIndex < handlerList.iterationCount; handlerList.iterationIndex++) {
                if (handlerList.handlers[handlerList.iterationIndex].apply(this, args ?? []) === false) {
                    result = false;
                }
            }
        }

        return result
    }

    emit(type: string, ...values: any[]) {
        let args = Array.prototype.slice.call(arguments, 1)

        // Sending the meta-event every time is quite expensive
        // let params = Array.prototype.slice.call(arguments)
        // this.emitArgs('event', params);

        return this.emitArgs(type, args) !== false;
    }

    private handlerList(handlers: Handler[]): HandlerList {
        return {
            handlers: handlers,
            iterationCount: 0,
            iterationIndex: 0
        }
    }
}
