export class TypedMatrix implements Iterable<Float64Array> {
  constructor(
    public readonly data: Float64Array,
    public readonly rows: number,
    public readonly cols: number,
  ) {
    const expected = rows * cols;
    if (data.length < expected) {
      throw new Error(
        `TypedMatrix: data length ${data.length} < rows*cols (${expected})`,
      );
    }
  }

  get length(): number {
    return this.rows;
  }

  get(i: number, j: number): number {
    return this.data[i * this.cols + j];
  }

  set(i: number, j: number, val: number): void {
    this.data[i * this.cols + j] = val;
  }

  row(i: number): Float64Array {
    return new Float64Array(
      this.data.buffer,
      this.data.byteOffset + i * this.cols * 8,
      this.cols,
    );
  }

  clone(): TypedMatrix {
    const d = new Float64Array(this.data.subarray(0, this.rows * this.cols));
    return new TypedMatrix(d, this.rows, this.cols);
  }

  sliceRow(start: number, end?: number): TypedMatrix {
    const s = Math.max(0, start);
    const e = end === undefined ? this.rows : Math.min(this.rows, end);
    const n = e - s;
    if (n <= 0) return new TypedMatrix(new Float64Array(0), 0, this.cols);
    const off = s * this.cols;
    const d = new Float64Array(this.data.subarray(off, off + n * this.cols));
    return new TypedMatrix(d, n, this.cols);
  }

  [Symbol.iterator](): Iterator<Float64Array> {
    let i = 0;
    const self = this;
    return {
      next(): IteratorResult<Float64Array> {
        if (i >= self.rows) return { done: true, value: undefined as any };
        return { done: false, value: self.row(i++) };
      },
    };
  }

  static zeros(rows: number, cols: number): TypedMatrix {
    return new TypedMatrix(new Float64Array(rows * cols), rows, cols);
  }

  static eye(n: number): TypedMatrix {
    const m = TypedMatrix.zeros(n, n);
    for (let i = 0; i < n; i++) m.data[i * n + i] = 1;
    return m;
  }

  static fromRows(rows: Float64Array[]): TypedMatrix {
    if (rows.length === 0) return new TypedMatrix(new Float64Array(0), 0, 0);
    const cols = rows[0].length;
    const data = new Float64Array(rows.length * cols);
    for (let i = 0; i < rows.length; i++) {
      data.set(rows[i], i * cols);
    }
    return new TypedMatrix(data, rows.length, cols);
  }

  static fromFloat32Arrays(rows: Float32Array[]): TypedMatrix {
    if (rows.length === 0) return new TypedMatrix(new Float64Array(0), 0, 0);
    const cols = rows[0].length;
    const data = new Float64Array(rows.length * cols);
    for (let i = 0; i < rows.length; i++) {
      for (let j = 0; j < cols; j++) {
        data[i * cols + j] = rows[i][j];
      }
    }
    return new TypedMatrix(data, rows.length, cols);
  }

  static fromNumberArrays(rows: number[][]): TypedMatrix {
    if (rows.length === 0) return new TypedMatrix(new Float64Array(0), 0, 0);
    const cols = rows[0].length;
    const data = new Float64Array(rows.length * cols);
    for (let i = 0; i < rows.length; i++) {
      for (let j = 0; j < cols; j++) {
        data[i * cols + j] = rows[i][j];
      }
    }
    return new TypedMatrix(data, rows.length, cols);
  }

  toFloat64Arrays(): Float64Array[] {
    const out: Float64Array[] = new Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      out[i] = this.row(i);
    }
    return out;
  }

  toNumberArrays(): number[][] {
    const out: number[][] = new Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      const row: number[] = new Array(this.cols);
      const off = i * this.cols;
      for (let j = 0; j < this.cols; j++) {
        row[j] = this.data[off + j];
      }
      out[i] = row;
    }
    return out;
  }
}
