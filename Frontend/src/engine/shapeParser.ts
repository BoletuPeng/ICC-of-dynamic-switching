/**
 * Shape Parser - Converts shape definitions to evaluable expressions
 *
 * Supports parsing:
 * - Numbers: 106, 242
 * - Variables: n_timepoints, n_rois
 * - Expressions: n_timepoints - window - 1, n_rois * n_rois
 * - Parameter references: @window (refers to node parameter 'window')
 */

import type { ShapeDimension, ShapeExpression } from './types';

// =============================================================================
// Tokenizer
// =============================================================================

type TokenType = 'number' | 'identifier' | 'operator' | 'lparen' | 'rparen' | 'paramref';

interface Token {
  type: TokenType;
  value: string | number;
}

/**
 * Tokenize a shape expression string
 */
function tokenize(expr: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  while (i < expr.length) {
    const ch = expr[i];

    // Skip whitespace
    if (/\s/.test(ch)) {
      i++;
      continue;
    }

    // Number
    if (/\d/.test(ch)) {
      let num = '';
      while (i < expr.length && /\d/.test(expr[i])) {
        num += expr[i++];
      }
      tokens.push({ type: 'number', value: parseInt(num, 10) });
      continue;
    }

    // Parameter reference (@param_name)
    if (ch === '@') {
      i++;
      let name = '';
      while (i < expr.length && /[a-zA-Z0-9_]/.test(expr[i])) {
        name += expr[i++];
      }
      tokens.push({ type: 'paramref', value: name });
      continue;
    }

    // Identifier
    if (/[a-zA-Z_]/.test(ch)) {
      let name = '';
      while (i < expr.length && /[a-zA-Z0-9_]/.test(expr[i])) {
        name += expr[i++];
      }
      tokens.push({ type: 'identifier', value: name });
      continue;
    }

    // Operators
    if (['+', '-', '*', '/'].includes(ch)) {
      tokens.push({ type: 'operator', value: ch });
      i++;
      continue;
    }

    // Parentheses
    if (ch === '(') {
      tokens.push({ type: 'lparen', value: '(' });
      i++;
      continue;
    }
    if (ch === ')') {
      tokens.push({ type: 'rparen', value: ')' });
      i++;
      continue;
    }

    throw new Error(`Unexpected character in shape expression: '${ch}' at position ${i}`);
  }

  return tokens;
}

// =============================================================================
// Parser (Recursive Descent)
// =============================================================================

class Parser {
  private tokens: Token[];
  private pos: number = 0;

  constructor(tokens: Token[]) {
    this.tokens = tokens;
  }

  private peek(): Token | undefined {
    return this.tokens[this.pos];
  }

  private consume(): Token {
    return this.tokens[this.pos++];
  }

  private expect(type: TokenType): Token {
    const token = this.consume();
    if (!token || token.type !== type) {
      throw new Error(`Expected ${type}, got ${token?.type || 'EOF'}`);
    }
    return token;
  }

  /**
   * Parse additive expression (+ -)
   */
  parseExpression(): ShapeDimension {
    let left = this.parseTerm();

    while (this.peek()?.type === 'operator' && ['+', '-'].includes(this.peek()!.value as string)) {
      const op = this.consume().value as string;
      const right = this.parseTerm();

      left = {
        op: op === '+' ? 'add' : 'sub',
        left,
        right,
      };
    }

    return left;
  }

  /**
   * Parse multiplicative expression (* /)
   */
  private parseTerm(): ShapeDimension {
    let left = this.parseFactor();

    while (this.peek()?.type === 'operator' && ['*', '/'].includes(this.peek()!.value as string)) {
      const op = this.consume().value as string;
      const right = this.parseFactor();

      left = {
        op: op === '*' ? 'mul' : 'div',
        left,
        right,
      };
    }

    return left;
  }

  /**
   * Parse primary expression (number, identifier, parenthesized expression)
   */
  private parseFactor(): ShapeDimension {
    const token = this.peek();

    if (!token) {
      throw new Error('Unexpected end of expression');
    }

    if (token.type === 'number') {
      this.consume();
      return token.value as number;
    }

    if (token.type === 'identifier') {
      this.consume();
      return token.value as string;
    }

    if (token.type === 'paramref') {
      this.consume();
      return { op: 'ref', param: token.value as string };
    }

    if (token.type === 'lparen') {
      this.consume();
      const expr = this.parseExpression();
      this.expect('rparen');
      return expr;
    }

    throw new Error(`Unexpected token: ${token.type}`);
  }
}

// =============================================================================
// Public API
// =============================================================================

/**
 * Parse a shape dimension from a string or pass through if already parsed
 */
export function parseShapeDimension(dim: string | number | ShapeExpression): ShapeDimension {
  // Already a number
  if (typeof dim === 'number') {
    return dim;
  }

  // Already an expression object
  if (typeof dim === 'object') {
    return dim;
  }

  // Simple identifier (no operators)
  if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(dim)) {
    return dim;
  }

  // Parse complex expression
  const tokens = tokenize(dim);
  if (tokens.length === 0) {
    throw new Error('Empty shape expression');
  }

  const parser = new Parser(tokens);
  return parser.parseExpression();
}

/**
 * Parse a complete shape definition
 */
export function parseShapeDefinition(shape: (string | number)[]): ShapeDimension[] {
  return shape.map(parseShapeDimension);
}

/**
 * Convert a ShapeDimension back to a human-readable string
 */
export function dimensionToString(dim: ShapeDimension): string {
  if (typeof dim === 'number') {
    return dim.toString();
  }

  if (typeof dim === 'string') {
    return dim;
  }

  // It's a ShapeExpression
  if (dim.op === 'ref') {
    return `@${dim.param}`;
  }

  const left = dimensionToString(dim.left!);
  const right = dimensionToString(dim.right!);

  const opSymbol = {
    add: '+',
    sub: '-',
    mul: '*',
    div: '/',
  }[dim.op];

  // Add parentheses for clarity in complex expressions
  const leftStr = typeof dim.left === 'object' && dim.left.op !== 'ref' ? `(${left})` : left;
  const rightStr = typeof dim.right === 'object' && dim.right.op !== 'ref' ? `(${right})` : right;

  return `${leftStr} ${opSymbol} ${rightStr}`;
}

/**
 * Convert a complete shape to string representation
 */
export function shapeToString(shape: ShapeDimension[]): string {
  return shape.map(dimensionToString).join(' × ');
}
