# Fix for Stage 3.5B Token Overflow Issues

## Problem Summary

Stage 3.5B (Method Benchmarking) was repeatedly failing with `max_tokens` errors:

```
Error code: 400 - {'error': {'message': "'max_tokens' or 'max_completion_tokens' is too large: 8192.
This model's maximum context length is 32768 tokens and your request has 25125 input tokens
(8192 > 32768 - 25125)."}}
```

### Root Causes

1. **Token Limit Configuration**: `SECONDARY_LLM_CONFIG` was set to `max_tokens: 8192`, which was too large when combined with growing conversation context
2. **Infinite Thinking Loops**: The agent was getting stuck in repetitive `<think>` blocks, repeating the same reasoning hundreds of times
3. **Context Bloat**: Verbose thinking was accumulating in the message history, causing exponential growth in context size

## Solutions Implemented

### 1. Reduced max_tokens Configuration (⚠️ REVERTED - NEEDS RE-APPLYING)

**File**: `conversational/code/config.py`

**NOTE**: This change was reverted. The config currently has:
```python
"max_tokens": 8192,  # Still at 8192 - needs reduction!
```

**Should be changed to**:
```python
"max_tokens": 4096,  # Reduced from 8192 to prevent context overflow with large inputs
```

**Rationale**: With 32768 token context and 25125 input tokens, only ~7600 tokens remain. Setting to 4096 provides adequate room for responses while preventing overflow.

**Action Required**: Manually change line 69 in config.py from 8192 to 4096.

### 2. Added Conciseness Instructions to System Prompts

**Files Modified**:
- `conversational/code/stage3_5b_agent.py`
- `conversational/code/stage3b_agent.py`
- `conversational/code/stage3_5a_agent.py`

Added to each system prompt:
```
## CRITICAL: Be Concise and Action-Oriented
❌ DO NOT write lengthy explanations or repetitive thinking
❌ DO NOT repeat the same reasoning multiple times
✅ Think briefly, then ACT immediately with tool calls
✅ Keep responses under 500 tokens
✅ Be direct and efficient
```

**Rationale**: Explicitly instructs the LLM to avoid verbose, repetitive thinking patterns.

### 3. Implemented <think> Tag Stripping

**Files Modified**:
- `conversational/code/stage3_5b_agent.py` (line 242-250)
- `conversational/code/stage3b_agent.py` (line 170-178)
- `conversational/code/stage3_5a_agent.py` (line 251-259)

Added in each `agent_node` function:
```python
# Strip verbose <think> tags to prevent context bloat
if response.content:
    import re
    # Remove <think>...</think> blocks to save context
    cleaned_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL)
    # If the cleaned content is empty but we have tool calls, provide minimal message
    if not cleaned_content.strip() and response.tool_calls:
        cleaned_content = "Executing tools..."
    response.content = cleaned_content.strip()
```

**Rationale**:
- Removes verbose thinking from message history
- Prevents context accumulation across iterations
- Preserves tool calls and essential messages
- Dramatically reduces token usage in multi-iteration agents

### 4. Limited Debug Logging Output

Changed debug logging from:
```python
logger.debug(f"Stage 3.5B Agent Response: {response.content}")
```

To:
```python
logger.debug(f"Stage 3.5B Agent Response: {response.content[:200]}...")  # Only log first 200 chars
```

**Rationale**: Prevents logs from being cluttered with massive repetitive thinking blocks.

## Expected Impact

1. **Reduced Token Overflow Errors**: max_tokens of 4096 leaves sufficient buffer for context growth
2. **Faster Execution**: Less verbose responses mean fewer tokens and faster completion
3. **Better Context Management**: Stripping `<think>` tags prevents exponential context growth
4. **Improved Success Rate**: The retry logic should now succeed on first or second attempt

## Testing Recommendations

1. Run Stage 3.5B again and monitor:
   - Token usage in logs
   - Success on first attempt vs retries
   - Total execution time

2. Check that:
   - Agents still produce correct tool calls
   - Method benchmarking completes successfully
   - Results are valid and consistent

3. If issues persist:
   - Further reduce max_tokens to 3072 or 2048
   - Increase STAGE_MAX_ROUNDS limits if agents need more iterations
   - Consider implementing conversation summarization

## Rollback Instructions

If these changes cause issues, revert:

1. In `config.py`: Change `max_tokens: 4096` back to `8192`
2. Remove the "Be Concise" sections from system prompts
3. Remove the `<think>` tag stripping code blocks
4. Restore full debug logging

## Files Modified

- `conversational/code/config.py` (line 69)
- `conversational/code/stage3_5b_agent.py` (lines 52-59, 242-250, 253)
- `conversational/code/stage3b_agent.py` (lines 52-57, 170-178, 181)
- `conversational/code/stage3_5a_agent.py` (lines 52-57, 251-259, 262)

---

**Date**: 2025-12-08
**Fixed By**: Claude Code Assistant
**Priority**: Critical - Prevents pipeline failures
