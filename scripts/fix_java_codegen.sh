#!/usr/bin/env bash
# Post-processing script for openapi-generator Java output.
#
# Fixes known codegen bugs when generating from OpenAPI 3.1 specs:
#   1. List<X>.class  → List.class  (Java type erasure forbids parameterized .class)
#   2. getList<X>()   → getListOfX() (angle brackets invalid in method names)
#   3. = [a, b, ...]; → new ArrayList<>() (raw array literals aren't valid Java)
#   4. Ctor(List<List<X>>) removed   (type erasure clash with Ctor(List<X>))
#   5. = bareword;    → ;            (unquoted enum defaults aren't valid Java)
#
# Usage: ./scripts/fix_java_codegen.sh clients/java/src

set -euo pipefail

dir="${1:?Usage: $0 <java-source-dir>}"

if [ ! -d "$dir" ]; then
    echo "Error: directory '$dir' does not exist" >&2
    exit 1
fi

count=0
while IFS= read -r -d '' file; do
    changed=false

    # 1. List<...>.class → List.class (all nesting levels)
    if grep -q '<[^>]*>\.class' "$file"; then
        perl -pi -e 's/\bList<[^>]*(?:<[^>]*>)*>\.class/List.class/g' "$file"
        changed=true
    fi

    # 2. Method names with angle brackets: getList<X>() → getListOfX()
    #    Handles nested generics like getList<List<Integer>>() → getListOfListInteger()
    if grep -q 'get\w*<[^(]*>(' "$file"; then
        perl -pi -e '
            # Only transform method declaration lines (public ... getX<Y>(...))
            s{(get\w*)<([^()]+)>(\s*\()}{
                my ($prefix, $inner, $suffix) = ($1, $2, $3);
                # Sanitize the generic content into a valid Java identifier
                $inner =~ s/[^A-Za-z0-9_]//g;
                "${prefix}Of${inner}${suffix}"
            }ge;
        ' "$file"
        changed=true
    fi

    # 3. Raw array literal defaults: = [a, b, ...]; → = new ArrayList<>();
    if grep -q ' = \[' "$file"; then
        perl -pi -e 's/ = \[[^\]]*\];/ = new ArrayList<>();/g' "$file"
        changed=true
    fi

    # 4. Remove List<List<X>> constructors that clash with List<X> constructors (type erasure)
    if grep -q '(List<List<' "$file"; then
        perl -0777 -pi -e '
            # Remove constructor blocks: public ClassName(List<List<...>> o) { ... }
            s/\n    public \w+\(List<List<[^>]*>> o\) \{\n        super\([^)]+\);\n        setActualInstance\(o\);\n    \}\n//g;
        ' "$file"
        changed=true
    fi

    # 5. Bare enum defaults: = http; → ; (unquoted identifiers as defaults)
    #    Only targets non-Java-keyword barewords assigned to enum-typed fields.
    if grep -qE 'private [A-Z]\w+ \w+ = [a-z][a-z_]*;' "$file"; then
        perl -pi -e 's/(private [A-Z]\w+ \w+) = (?!null;|true;|false;|new )[a-z][a-z_]*;/$1;/g' "$file"
        changed=true
    fi

    if [ "$changed" = true ]; then
        count=$((count + 1))
    fi
done < <(find "$dir" -name '*.java' -print0)

echo "Post-processed $count Java file(s) in $dir"
