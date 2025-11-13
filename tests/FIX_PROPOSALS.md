# Proposed Fixes for OrderDataQueue Busy-Waiting Issue

## Problem Summary
When `shuffle=False`, `OrderDataQueue.put()` returns `False` when it's not a thread's turn, causing orchestration threads to spin with `sleep(0.0001)`, leading to high CPU usage and apparent slowness.

## Solution Options

### Option 1: Use `threading.Condition` for Proper Blocking ⭐ (RECOMMENDED)

**Concept**: Replace busy-waiting with proper blocking using `threading.Condition` that notifies threads when it's their turn.

**Implementation**:
- Add `threading.Condition` to `OrderDataQueue`
- When `put()` fails (not thread's turn or queue full), thread waits on condition
- When `get()` is called or `update_indices()` changes `cur_thread_idx`, notify waiting threads
- Threads wake up and check if it's their turn

**Pros**:
- ✅ Eliminates CPU spinning
- ✅ Proper synchronization primitive
- ✅ Threads block efficiently until ready
- ✅ Standard Python pattern

**Cons**:
- ⚠️ Requires careful lock management
- ⚠️ Need to ensure all wait paths are notified

**Code Changes**:
```python
# In OrderDataQueue.__init__:
self.condition = threading.Condition(self.lock)  # Reuse existing lock

# In OrderDataQueue.put():
with self.lock:
    if self.cur_thread_idx != thread_index:
        # Wait until it's our turn
        while self.cur_thread_idx != thread_index:
            self.condition.wait()  # Blocks until notified
        # Continue with put logic...

# In OrderDataQueue.update_indices():
# After updating cur_thread_idx:
self.condition.notify_all()  # Wake up waiting threads

# In OrderDataQueue.get():
# After removing item:
self.condition.notify_all()  # Wake up threads waiting for queue space
```

---

### Option 2: Adaptive Exponential Backoff

**Concept**: Increase sleep time when `put()` repeatedly fails, reducing CPU usage.

**Implementation**:
- Track consecutive failures in orchestration thread
- Start with `sleep(0.0001)`, double on each failure up to a max (e.g., `0.01s`)
- Reset to minimum when `put()` succeeds

**Pros**:
- ✅ Simple to implement
- ✅ Minimal code changes
- ✅ Reduces CPU usage significantly

**Cons**:
- ⚠️ Still uses polling (less efficient than blocking)
- ⚠️ Adds latency when thread becomes ready
- ⚠️ Not as elegant as proper synchronization

**Code Changes**:
```python
# In OrchestrateSingleWorker:
consecutive_failures = 0
max_sleep = 0.01
min_sleep = 0.0001

while not self.data_queue.put(data, self.thread_index):
    sleep_time = min(min_sleep * (2 ** consecutive_failures), max_sleep)
    time.sleep(sleep_time)
    consecutive_failures += 1
    if self.close_event.is_set():
        return

# On success:
consecutive_failures = 0
```

---

### Option 3: Per-Thread Condition Variables

**Concept**: Each thread has its own condition variable, only notify the specific thread whose turn it is.

**Implementation**:
- Maintain a list of `threading.Condition` objects, one per thread
- When `cur_thread_idx` changes, notify only that thread's condition
- Threads wait on their specific condition

**Pros**:
- ✅ More efficient (only wakes relevant thread)
- ✅ Eliminates unnecessary wake-ups
- ✅ Better scalability

**Cons**:
- ⚠️ More complex to implement
- ⚠️ Need to manage multiple condition objects
- ⚠️ Must ensure proper cleanup

**Code Changes**:
```python
# In OrderDataQueue.__init__:
self.thread_conditions = [threading.Condition(self.lock) for _ in range(nb_thread)]

# In OrderDataQueue.put():
if self.cur_thread_idx != thread_index:
    while self.cur_thread_idx != thread_index:
        self.thread_conditions[thread_index].wait()

# In OrderDataQueue.update_indices():
# After updating cur_thread_idx:
next_thread = self.cur_thread_idx
self.thread_conditions[next_thread].notify()  # Only wake next thread
```

---

### Option 4: Make `put()` Blocking with Timeout

**Concept**: Change `put()` to block until success or timeout, eliminating the need for external loops.

**Implementation**:
- `put()` internally waits using condition variable
- Add timeout parameter for safety
- Return success/failure after blocking

**Pros**:
- ✅ Encapsulates waiting logic
- ✅ Cleaner orchestration thread code
- ✅ Can add timeout for safety

**Cons**:
- ⚠️ Changes API (but backward compatible if timeout is optional)
- ⚠️ Need to handle timeouts properly

**Code Changes**:
```python
# In OrderDataQueue.put():
def put(self, item: Any, thread_index: int, timeout=None):
    with self.lock:
        # Wait until it's our turn
        if self.cur_thread_idx != thread_index:
            deadline = time.time() + timeout if timeout else None
            while self.cur_thread_idx != thread_index:
                if timeout and time.time() > deadline:
                    return False
                self.condition.wait(timeout=0.1)  # Check periodically
        
        # Continue with existing put logic...
```

---

### Option 5: Separate Queues Per Thread (Major Refactor)

**Concept**: Each thread has its own queue, consumer reads from queues in round-robin order.

**Implementation**:
- Replace single `OrderDataQueue` with list of per-thread queues
- Orchestration threads put into their own queue (no blocking)
- Consumer reads from queues in round-robin order

**Pros**:
- ✅ Eliminates ordering bottleneck
- ✅ No waiting for "turn"
- ✅ Better parallelism

**Cons**:
- ⚠️ Major architectural change
- ⚠️ More complex consumer logic
- ⚠️ Need to handle empty queues
- ⚠️ May break existing behavior

---

## Recommended Approach: **Option 1 (threading.Condition)**

**Why**:
1. **Standard Solution**: Uses proper Python synchronization primitives
2. **Efficient**: Threads block instead of spinning, saving CPU
3. **Minimal Changes**: Only affects `OrderDataQueue` class
4. **Backward Compatible**: Doesn't change external API
5. **Well-Tested Pattern**: Condition variables are a standard solution for this problem

**Implementation Strategy**:
1. Add `threading.Condition` to `OrderDataQueue.__init__()`
2. Modify `put()` to wait on condition when not thread's turn or queue full
3. Modify `update_indices()` to notify waiting threads when turn changes
4. Modify `get()` to notify waiting threads when space becomes available
5. Test thoroughly to ensure no deadlocks

**Additional Considerations**:
- Ensure `close_event` checking still works (check before/after wait)
- Handle edge cases (thread termination, queue closing)
- Consider adding timeout to `condition.wait()` for safety
- Test with various worker counts and queue sizes

---

## Quick Win: Option 2 (Adaptive Backoff) as Interim Fix

If Option 1 requires more testing, Option 2 can be implemented quickly as an interim solution:
- Very simple change
- Immediate CPU usage reduction
- Can be replaced with Option 1 later
- Low risk

---

## Testing Strategy

After implementing any fix:
1. Test with `shuffle=False`, multiple workers
2. Monitor CPU usage (should be low)
3. Verify no deadlocks
4. Test with various configurations (1-1, 1-2, 2-2)
5. Test with slow/fast consumers
6. Test epoch transitions
7. Test cleanup/shutdown

