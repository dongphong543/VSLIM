#!/usr/bin/env python3
"""
Data validation script for VSLIM dataset.
Validates that:
1. Length of words in seq_in matches length of tokens in seq_intent_out and seq_out
2. If a token in seq_intent_out is 'O' and in seq_out is not 'O' (or vice versa), it's invalid
"""

import os
import sys
from typing import List, Tuple, Dict


def read_file_lines(file_path: str) -> List[str]:
    """Read all lines from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def validate_data_consistency(seq_in_lines: List[str], 
                            seq_intent_out_lines: List[str], 
                            seq_out_lines: List[str]) -> Dict:
    """
    Validate data consistency according to the specified rules.
    
    Returns:
        Dict with validation results including errors and statistics
    """
    results = {
        'total_lines': len(seq_in_lines),
        'valid_lines': 0,
        'invalid_lines': 0,
        'errors': [],
        'line_errors': []
    }
    
    # Check if all files have the same number of lines
    if len(seq_in_lines) != len(seq_intent_out_lines) or len(seq_in_lines) != len(seq_out_lines):
        results['errors'].append(f"Line count mismatch: seq_in={len(seq_in_lines)}, "
                               f"seq_intent_out={len(seq_intent_out_lines)}, "
                               f"seq_out={len(seq_out_lines)}")
        return results
    
    for line_idx in range(len(seq_in_lines)):
        line_num = line_idx + 1
        seq_in_line = seq_in_lines[line_idx]
        seq_intent_out_line = seq_intent_out_lines[line_idx]
        seq_out_line = seq_out_lines[line_idx]
        
        # Split into tokens
        seq_in_words = seq_in_line.split()
        seq_intent_tokens = seq_intent_out_line.split()
        seq_out_tokens = seq_out_line.split()
        
        line_errors = []
        
        # Check 1: Length consistency
        if len(seq_in_words) != len(seq_intent_tokens):
            line_errors.append(f"Word count mismatch: seq_in has {len(seq_in_words)} words, "
                           f"seq_intent_out has {len(seq_intent_tokens)} tokens")
        
        if len(seq_in_words) != len(seq_out_tokens):
            line_errors.append(f"Word count mismatch: seq_in has {len(seq_in_words)} words, "
                             f"seq_out has {len(seq_out_tokens)} tokens")
        
        if len(seq_intent_tokens) != len(seq_out_tokens):
            line_errors.append(f"Token count mismatch: seq_intent_out has {len(seq_intent_tokens)} tokens, "
                             f"seq_out has {len(seq_out_tokens)} tokens")
        
        # Check 2: O token consistency
        if len(seq_intent_tokens) == len(seq_out_tokens):
            for token_idx in range(len(seq_intent_tokens)):
                intent_token = seq_intent_tokens[token_idx]
                out_token = seq_out_tokens[token_idx]
                
                # If intent is O but out is not O, or vice versa
                if (intent_token == 'O' and out_token != 'O') or (intent_token != 'O' and out_token == 'O'):
                    line_errors.append(f"O token inconsistency at position {token_idx + 1}: "
                                     f"intent='{intent_token}', out='{out_token}'")
        
        # Check 3: I-type token validation (must follow B-type of same entity)
        if len(seq_out_tokens) > 0:
            for token_idx in range(len(seq_out_tokens)):
                current_token = seq_out_tokens[token_idx]
                
                # Check if current token is I-type
                if current_token.startswith('I-'):
                    entity_type = current_token[2:]  # Remove 'I-' prefix
                    expected_b_token = f'B-{entity_type}'
                    
                    # Check if previous token is the expected B-type
                    if token_idx == 0:
                        # I-token at the beginning of sequence is invalid
                        line_errors.append(f"I-type token '{current_token}' at position {token_idx + 1} "
                                         f"cannot start a sequence (must follow B-{entity_type})")
                    else:
                        previous_token = seq_out_tokens[token_idx - 1]
                        if previous_token != expected_b_token and previous_token != current_token:
                            # Previous token is not the expected B-type or same I-type
                            line_errors.append(f"I-type token '{current_token}' at position {token_idx + 1} "
                                             f"does not follow expected B-{entity_type} or I-{entity_type} "
                                             f"(previous token: '{previous_token}')")
        
        # Check 4: Intent-token type compatibility
        if len(seq_intent_tokens) == len(seq_out_tokens):
            # Define intent-token type compatibility rules
            intent_token_rules = {
                'add_transaction': ['target'],  # target only
                'update_transaction': ['target', 'condition'],  # target or condition
                'delete_transaction': ['condition'],  # condition only
                'stat_transaction': ['condition'],  # condition only
                'search_transaction': ['condition'],  # condition only
                'O': ['target', 'condition']  # O can have any token type
            }
            
            for token_idx in range(len(seq_intent_tokens)):
                intent_token = seq_intent_tokens[token_idx]
                out_token = seq_out_tokens[token_idx]
                
                # Skip O tokens in intent (they can have any token type)
                if intent_token == 'O':
                    continue
                
                # Check if intent token has defined rules
                if intent_token in intent_token_rules:
                    allowed_types = intent_token_rules[intent_token]
                    
                    if out_token.startswith('B-') or out_token.startswith('I-'):
                        token_type = out_token.split('-')[1].split('_')[0]  # Extract type after B- or I-
                        
                        if token_type not in allowed_types:
                            line_errors.append(f"Intent '{intent_token}' at position {token_idx + 1} "
                                             f"is incompatible with token '{out_token}' "
                                             f"(allowed types: {', '.join(allowed_types)})")
                    else:
                        # Unknown token format
                        line_errors.append(f"Unknown token format '{out_token}' at position {token_idx + 1}")
        
        if line_errors:
            results['invalid_lines'] += 1
            results['line_errors'].append({
                'line_number': line_num,
                'errors': line_errors,
                'seq_in': seq_in_line,
                'seq_intent_out': seq_intent_out_line,
                'seq_out': seq_out_line
            })
        else:
            results['valid_lines'] += 1
    
    return results


def print_validation_results(results: Dict):
    """Print validation results in a formatted way."""
    print("=" * 80)
    print("DATA VALIDATION RESULTS")
    print("=" * 80)
    
    print(f"Total lines processed: {results['total_lines']}")
    print(f"Valid lines: {results['valid_lines']}")
    print(f"Invalid lines: {results['invalid_lines']}")
    
    if results['errors']:
        print("\nGENERAL ERRORS:")
        for error in results['errors']:
            print(f"  - {error}")
    
    if results['line_errors']:
        print(f"\nINVALID LINES ({len(results['line_errors'])} lines with errors):")
        print("=" * 80)
        
        for line_error in results['line_errors']:
            print(f"\nLine {line_error['line_number']}:")
            for error in line_error['errors']:
                print(f"  ❌ {error}")
            print(f"  📝 seq_in: {line_error['seq_in']}")
            print(f"  🏷️  seq_intent_out: {line_error['seq_intent_out']}")
            print(f"  🎯 seq_out: {line_error['seq_out']}")
            print("-" * 80)
    
    print("\n" + "=" * 80)
    
    if results['invalid_lines'] == 0:
        print("✅ ALL DATA IS VALID!")
    else:
        print(f"❌ {results['invalid_lines']} lines have validation errors")
        print("\nSUMMARY OF INVALID LINES:")
        for line_error in results['line_errors']:
            print(f"Line {line_error['line_number']}: {', '.join(line_error['errors'])}")


def fix_wrong_labels(seq_intent_out_lines: List[str], seq_out_lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    Fix wrong labels based on intent-token compatibility rules.
    Returns fixed versions of seq_intent_out and seq_out.
    """
    # Define intent-token type compatibility rules
    intent_token_rules = {
        'add_transaction': ['target'],  # target only
        # 'update_transaction': ['target', 'condition'],  # target or condition (unchanged)
        'delete_transaction': ['condition'],  # condition only
        'stat_transaction': ['condition'],  # condition only
        'search_transaction': ['condition'],  # condition only
        'O': ['target', 'condition']  # O can have any token type
    }
    
    fixed_intent_lines = []
    fixed_out_lines = []
    fixes_applied = 0
    
    for line_idx in range(len(seq_intent_out_lines)):
        intent_tokens = seq_intent_out_lines[line_idx].split()
        out_tokens = seq_out_lines[line_idx].split()
        
        fixed_intent_tokens = intent_tokens.copy()
        fixed_out_tokens = out_tokens.copy()
        
        # Fix intent-token compatibility issues
        for token_idx in range(len(intent_tokens)):
            intent_token = intent_tokens[token_idx]
            out_token = out_tokens[token_idx]
            
            # Skip O tokens in intent (they can have any token type)
            if intent_token == 'O':
                continue
            
            # Check if intent token has defined rules
            if intent_token in intent_token_rules:
                allowed_types = intent_token_rules[intent_token]
                
                if out_token.startswith('B-') or out_token.startswith('I-'):
                    token_type = out_token.split('-')[1]  # Extract type after B- or I-
                    
                    if token_type not in allowed_types:
                        # Fix the token type based on intent
                        if intent_token == 'add_transaction':
                            # Change to target type - map condition tokens to target tokens
                            if out_token.startswith('B-'):
                                if token_type == 'condition_description':
                                    fixed_out_tokens[token_idx] = 'B-target_description'
                                elif token_type == 'condition_price':
                                    fixed_out_tokens[token_idx] = 'B-target_price'
                                elif token_type == 'condition_date':
                                    fixed_out_tokens[token_idx] = 'B-target_date'
                                elif token_type == 'condition_location':
                                    fixed_out_tokens[token_idx] = 'B-target_location'
                            elif out_token.startswith('I-'):
                                if token_type == 'condition_description':
                                    fixed_out_tokens[token_idx] = 'I-target_description'
                                elif token_type == 'condition_price':
                                    fixed_out_tokens[token_idx] = 'I-target_price'
                                elif token_type == 'condition_date':
                                    fixed_out_tokens[token_idx] = 'I-target_date'
                                elif token_type == 'condition_location':
                                    fixed_out_tokens[token_idx] = 'I-target_location'
                        elif intent_token in ['delete_transaction', 'stat_transaction', 'search_transaction']:
                            # Change to condition type - map target tokens to condition tokens
                            if out_token.startswith('B-'):
                                if token_type == 'target_description':
                                    fixed_out_tokens[token_idx] = 'B-condition_description'
                                elif token_type == 'target_price':
                                    fixed_out_tokens[token_idx] = 'B-condition_price'
                                elif token_type == 'target_date':
                                    fixed_out_tokens[token_idx] = 'B-condition_date'
                                elif token_type == 'target_location':
                                    fixed_out_tokens[token_idx] = 'B-condition_location'
                            elif out_token.startswith('I-'):
                                if token_type == 'target_description':
                                    fixed_out_tokens[token_idx] = 'I-condition_description'
                                elif token_type == 'target_price':
                                    fixed_out_tokens[token_idx] = 'I-condition_price'
                                elif token_type == 'target_date':
                                    fixed_out_tokens[token_idx] = 'I-condition_date'
                                elif token_type == 'target_location':
                                    fixed_out_tokens[token_idx] = 'I-condition_location'
                        
                        fixes_applied += 1
        
        fixed_intent_lines.append(' '.join(fixed_intent_tokens))
        fixed_out_lines.append(' '.join(fixed_out_tokens))
    
    print(f"Applied {fixes_applied} fixes to labels")
    return fixed_intent_lines, fixed_out_lines


def save_fixed_files(seq_intent_out_lines: List[str], seq_out_lines: List[str], data_dir: str):
    """Save the fixed lines back to files."""
    seq_intent_out_path = os.path.join(data_dir, "seq_intent_out.txt")
    seq_out_path = os.path.join(data_dir, "seq_out.txt")
    
    # Create backup files
    backup_intent_path = seq_intent_out_path + ".backup"
    backup_out_path = seq_out_path + ".backup"
    
    # Backup original files
    import shutil
    shutil.copy2(seq_intent_out_path, backup_intent_path)
    shutil.copy2(seq_out_path, backup_out_path)
    print(f"Created backups: {backup_intent_path}, {backup_out_path}")
    
    # Save fixed files
    with open(seq_intent_out_path, 'w', encoding='utf-8') as f:
        for line in seq_intent_out_lines:
            f.write(line + '\n')
    
    with open(seq_out_path, 'w', encoding='utf-8') as f:
        for line in seq_out_lines:
            f.write(line + '\n')
    
    print(f"Saved fixed files: {seq_intent_out_path}, {seq_out_path}")


def get_invalid_lines(results: Dict) -> List[Dict]:
    """Return a list of invalid lines with their details."""
    return results['line_errors']


def main():
    """Main function to run data validation and fixing."""
    # Define file paths
    data_dir = "data/vped/train"
    seq_in_path = os.path.join(data_dir, "seq_in.txt")
    seq_intent_out_path = os.path.join(data_dir, "seq_intent_out.txt")
    seq_out_path = os.path.join(data_dir, "seq_out.txt")
    
    print("Loading data files...")
    
    # Read all files
    seq_in_lines = read_file_lines(seq_in_path)
    seq_intent_out_lines = read_file_lines(seq_intent_out_path)
    seq_out_lines = read_file_lines(seq_out_path)
    
    if not seq_in_lines or not seq_intent_out_lines or not seq_out_lines:
        print("Error: Could not read one or more data files")
        sys.exit(1)
    
    print(f"Loaded {len(seq_in_lines)} lines from each file")
    print("Starting label fixing...")
    
    # Fix wrong labels
    fixed_intent_lines, fixed_out_lines = fix_wrong_labels(seq_intent_out_lines, seq_out_lines)
    
    # Save fixed files
    save_fixed_files(fixed_intent_lines, fixed_out_lines, data_dir)
    
    print("Label fixing completed!")
    
    # Now validate the fixed data
    print("\nValidating fixed data...")
    results = validate_data_consistency(seq_in_lines, fixed_intent_lines, fixed_out_lines)
    print_validation_results(results)
    
    # Exit with appropriate code
    if results['invalid_lines'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
