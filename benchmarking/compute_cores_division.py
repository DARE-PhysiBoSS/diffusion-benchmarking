

def integer_triples(X: int):
    triples = []
    for a in range(1, X + 1):
        if X % a != 0:
            continue
        for b in range(1, X + 1):
            if (X // a) % b != 0:
                continue
            c = X // (a * b)

            triples.append([a, b, c])
    return triples


if __name__ == "__main__":
    X = int(input("Enter a positive integer X: "))
    result = integer_triples(X)
    print(f"All positive integer triples (a, b, c) such that a*b*c = {X}:")
    print(", ".join(str(triple) for triple in result))