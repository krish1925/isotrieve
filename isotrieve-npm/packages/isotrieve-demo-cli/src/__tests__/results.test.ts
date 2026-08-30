import { buildResultsTable } from '../results';

describe('buildResultsTable', () => {
    it('has the expected three-column header', () => {
        const table = buildResultsTable(10, 1);
        const rendered = table.toString();
        expect(rendered).toContain('Metric');
        expect(rendered).toContain('Text Handoff');
        expect(rendered).toContain('Isotrieve Vector Handoff');
    });

    it('formats both latencies to two decimal places', () => {
        const rendered = buildResultsTable(12.3456, 0.9876).toString();
        expect(rendered).toContain('12.35ms');
        expect(rendered).toContain('0.99ms');
    });

    it('includes the privacy row comparing text exposure vs vectors only', () => {
        const rendered = buildResultsTable(10, 1).toString();
        expect(rendered).toContain('Privacy');
        expect(rendered).toContain('Text Exposed');
        expect(rendered).toContain('Vectors Only');
    });

    it('includes the cost row comparing re-encoding vs matrix multiply', () => {
        const rendered = buildResultsTable(10, 1).toString();
        expect(rendered).toContain('Cost');
        expect(rendered).toContain('$$$ (Re-encode)');
        expect(rendered).toContain('FREE');
    });

    it('renders three metric rows', () => {
        const table = buildResultsTable(10, 1);
        expect(table.length).toBe(3);
    });

    it('handles zero-latency edge cases without throwing', () => {
        expect(() => buildResultsTable(0, 0)).not.toThrow();
        const rendered = buildResultsTable(0, 0).toString();
        expect(rendered).toContain('0.00ms');
    });
});
