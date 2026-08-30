import Table from 'cli-table3';
import chalk from 'chalk';

/**
 * Build the side-by-side results table comparing a traditional
 * text handoff (re-encoding) against an Isotrieve vector handoff
 * (matrix multiply in the target embedding space).
 */
export function buildResultsTable(textMs: number, isotrieveMs: number) {
    const table = new Table({
        head: ['Metric', 'Text Handoff', 'Isotrieve Vector Handoff'],
        style: { head: ['cyan'] }
    });

    table.push(
        ['Latency', chalk.red(`${textMs.toFixed(2)}ms`), chalk.green(`${isotrieveMs.toFixed(2)}ms`)],
        ['Privacy', chalk.red('Text Exposed'), chalk.green('Vectors Only')],
        ['Cost', chalk.red('$$$ (Re-encode)'), chalk.green('FREE')]
    );

    return table;
}
