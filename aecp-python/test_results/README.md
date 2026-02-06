# Incremental Calibration Test Results

## Files Generated

1. **`incremental_calibration_20260204_1507.json`** - Complete test data
   - 90 steps (every 500 words from 500 to 45,000)
   - Validation similarities at each step
   - Training similarities at each step
   - Step times and cumulative times

2. **`incremental_calibration_graphs.png`** - Main visualization
   - 4 plots showing:
     - Validation similarity vs vocabulary size
     - Step time vs vocabulary size (explains why time increases)
     - Cumulative training time
     - Training vs validation similarity comparison

3. **`step_time_analysis.png`** - Detailed step time analysis
   - Shows linear relationship between vocabulary size and step time
   - Explains why each step takes longer

## Why Step Time Increases

**This is EXPECTED behavior!**

Each step processes MORE vocabulary than the previous step:
- Step 1: Encodes 500 words → ~1-2 seconds
- Step 2: Encodes 1,000 words → ~2-3 seconds  
- Step 3: Encodes 1,500 words → ~3-4 seconds
- Step N: Encodes N×500 words → ~N×0.01-0.02 seconds

**Time complexity: O(vocab_size)**

The time increases linearly because:
1. Each step encodes ALL vocabulary up to that point (not just new words)
2. Matrix computation time increases with vocabulary size
3. Validation evaluation processes the same validation set each time

This is normal and expected - larger vocabulary = more computation time.

## Results Summary

- **Total steps**: 90
- **Total vocabulary tested**: 45,000 words
- **Validation set**: 5,000 words (held-out)
- **Final validation similarity**: 1.0000 (perfect - MockAdapter is deterministic)
- **Total time**: ~15 minutes (907 seconds)

## Key Insights

1. **Validation similarity**: Remains stable at 1.0 throughout (MockAdapter)
2. **Step time**: Increases linearly with vocabulary size (expected)
3. **Cumulative time**: Grows quadratically (sum of increasing step times)
4. **Efficiency**: Each step validates on the same 5K validation set

## Next Steps

For real embedding models (not MockAdapter):
- Expect validation similarity to improve with more vocabulary
- Step times will be longer (real encoding takes more time)
- Can optimize by only encoding NEW vocabulary each step (not re-encoding all)
