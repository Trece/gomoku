
def absearch(board, threshold, depth):
    movelist = good_moves(board)
    current_best = -1.
    for move in movelist:
        output = absearch(aftermove(board, move), max(threshold, current_best), depth)
