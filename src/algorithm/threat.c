#include "threat.h"

#include "board.h"
#include "eval.h"
#include "pattern.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>

void free_threat_info(void* ptr) {
    if (!ptr) return;
    threat_info_t* threat_info = ptr;
    vector_free(&threat_info->consists);
    vector_free(&threat_info->defenses);
}

threat_info_t clone_threat_info(threat_info_t* threat_info) {
    threat_info_t result = {
        .type = threat_info->type,
        .action = threat_info->action,
        .dx = threat_info->dx,
        .dy = threat_info->dy,
    };
    result.consists = vector_clone(threat_info->consists);
    result.defenses = vector_clone(threat_info->defenses);
    return result;
}

vector_t scan_threats_info(board_t board, int id, bool only_four) {
    vector_t threat_info = vector_new(threat_info_t, free_threat_info);
    comp_board_t cpboard;
    encode(board, cpboard);
    vector_t threats = vector_new(threat_t, NULL);
    threat_storage_t storage = {0};
    storage[PAT_WIN] = storage[PAT_A4] = storage[PAT_D4] = &threats;
    if (!only_four) {
        storage[PAT_A3] = &threats;
    }
    scan_threats(cpboard, id, storage);
    for_each(threat_t, threats, threat) {
        threat_info_t info = {
            .type = threat.pattern,
            .consists = {0},
            .defenses = {0},
            .action = threat.pos,
            .dx = threat.dx,
            .dy = threat.dy,
        };
        board[threat.pos.x][threat.pos.y] = id;
        info.consists = find_relative_points(CONSIST, board, threat.pos, threat.dx, threat.dy);
        info.defenses = find_relative_points(DEFEND, board, threat.pos, threat.dx, threat.dy);
        board[threat.pos.x][threat.pos.y] = 0;
        vector_push_back(threat_info, info);
    }
    vector_free(&threats);
    return threat_info;
}

vector_t find_threats(board_t board, point_t pos, bool only_four) {
    int id = board[pos.x][pos.y];
    vector_t result = vector_new(threat_info_t, free_threat_info);
    if (!id) return result;
    for_all_dir(d, dx, dy) {
        segment_t seg = get_segment(board, pos, dx, dy);
        int value = encode_segment(seg);
        pattern_t pat = to_upgraded_pattern(value, id == 1);
        bool select = false;
        if (only_four) {
            select = pat >= PAT_D4 && pat <= PAT_WIN;
        } else {
            select = pat >= PAT_A3 && pat <= PAT_WIN;
        }
        if (select) {
            int columns[5];
            get_upgrade_columns(value, id == 1, columns, 5);
            for (int i = 0; i < 5; i++) {
                if (columns[i] != -1) {
                    point_t np = column_to_point(pos, dx, dy, columns[i]);
                    if (!is_forbidden(board, np, id, false)) {
                        board[np.x][np.y] = id;
                        threat_info_t threat_info = {
                            .type = pat,
                            .action = np,
                            .consists = find_relative_points(CONSIST, board, np, dx, dy),
                            .defenses = find_relative_points(DEFEND, board, np, dx, dy),
                            .dx = dx,
                            .dy = dy,
                        };
                        board[np.x][np.y] = 0;
                        vector_push_back(result, threat_info);
                    }
                }
            }
        }
    }
    return result;
}

static int indent;

void free_tree_node(threat_tree_node_t* node) { free_threat_info(&node->threat); }

int delete_tree(threat_tree_node_t* root) {
    if (!root) return 0;
    int cnt = 1;
    for (threat_tree_node_t *child = root->son, *next; child; child = next) {
        next = child->brother;
        cnt += delete_tree(child);
    }
    free_tree_node(root);
    free(root);
    return cnt;
}

void free_threat_tree(void* ptr) {
    threat_tree_node_t** pnode = ptr;
    threat_tree_node_t* node = *pnode;
    delete_tree(node);
    *pnode = NULL;
}

#define READABLE_POS(pos) (pos.y + 'A'), (pos.x + 1)

void print_threat(threat_info_t threat) {
    char indent_str[256] = {0};
    for (int i = 0; i < indent; i++) {
        strcat(indent_str, "  ");
    }
    point_t pos = threat.action;
    log_l("%s%c%d: %s", indent_str, READABLE_POS(pos), pattern_typename[threat.type]);
}

void print_threat_tree(threat_tree_node_t* root) {
    print_threat(root->threat);
    indent++;
    for (threat_tree_node_t* child = root->son; child; child = child->brother) {
        print_threat_tree(child);
    }
    indent--;
}

static int start_time, TIME_LIMIT, current_depth, DEPTH_LIMIT;

static int offend_id, defend_id;

static board_t initial;

void act_round(board_t board, threat_info_t threat) {
    board[threat.action.x][threat.action.y] = offend_id;
    for_each(point_t, threat.defenses, defend) { board[defend.x][defend.y] = defend_id; }
}

void revert_round(board_t board, threat_info_t threat) {
    board[threat.action.x][threat.action.y] = 0;
    for_each(point_t, threat.defenses, defend) { board[defend.x][defend.y] = 0; }
}

void build_threat_tree(threat_tree_node_t* node);

void add_threat(threat_tree_node_t* node, threat_info_t threat) {
    threat_tree_node_t* child = malloc(sizeof(threat_tree_node_t));
    memcpy(child->board, node->board, sizeof(board_t));
    act_round(child->board, threat);
    child->threat = clone_threat_info(&threat);
    child->parent = node;
    child->son = NULL;
    child->brother = node->son;
    child->only_four = node->only_four;
    node->son = child;
    build_threat_tree(child);
}

void build_threat_tree(threat_tree_node_t* node) {
    // if (get_time(start_time) > time_limit) return;
    if (current_depth >= DEPTH_LIMIT) return;
    current_depth += 1;
    vector_t threats = find_threats(node->board, node->threat.action, node->only_four);
    for_each(threat_info_t, threats, threat) { add_threat(node, threat); }
    current_depth -= 1;
}

void record_sequence(threat_tree_node_t* node, vector_t* sequence) {
    if (node->parent) record_sequence(node->parent, sequence);
    vector_push_back(*sequence, node->threat);
}

void record_point_sequence(threat_tree_node_t* node, vector_t* sequence) {
    if (node->parent) record_point_sequence(node->parent, sequence);
    vector_push_back(*sequence, node->threat.action);
}

void print_sequence(vector_t sequence) {
    for_each(threat_info_t, sequence, threat) {
        print_threat(threat);
        act_round(initial, threat);
    }
    print(initial);
    for_each(threat_info_t, sequence, threat) { revert_round(initial, threat); }
}

bool have_same_point(vector_t points1, vector_t points2) {
    for_each(point_t, points1, point1) {
        for_each(point_t, points2, point2) {
            if (point1.x == point2.x && point1.y == point2.y) {
                return true;
            }
        }
    }
    return false;
}

bool compatible(threat_tree_node_t* node1, threat_tree_node_t* node2) {
    vector_t seq1 = vector_new(threat_info_t, NULL), seq2 = vector_new(threat_info_t, NULL);
    record_sequence(node1, &seq1);
    record_sequence(node2, &seq2);
    vector_t actions1 = vector_new(point_t, NULL), actions2 = vector_new(point_t, NULL);
    vector_t defense1 = vector_new(point_t, NULL), defense2 = vector_new(point_t, NULL);
    for_each(threat_info_t, seq1, threat) {
        vector_push_back(actions1, threat.action);
        vector_cat(defense1, threat.defenses);
    }
    for_each(threat_info_t, seq2, threat) {
        vector_push_back(actions2, threat.action);
        vector_cat(defense2, threat.defenses);
    }
    bool result = true;
    if (result && have_same_point(actions1, actions2)) result = false;
    if (result && have_same_point(actions1, defense2)) result = false;
    if (result && have_same_point(actions2, defense1)) result = false;
    if (result && have_same_point(defense1, defense2)) result = false;
    vector_free(&seq1), vector_free(&seq2);
    vector_free(&actions1), vector_free(&actions2);
    vector_free(&defense1), vector_free(&defense2);
    return result;
}

bool verify_sequence(vector_t sequence) {
    vector_t defenses = vector_new(threat_t, NULL);
    comp_board_t cpboard;
    bool result = true;
    for_each(threat_info_t, sequence, threat) {
        act_round(initial, threat);
        defenses.size = 0;
        threat_storage_t storage = {
            [PAT_WIN] = &defenses,
        };
        encode(initial, cpboard);
        scan_threats(cpboard, defend_id, storage);
        if (defenses.size) {
            result = false;
        }
    }
    for_each(threat_info_t, sequence, threat) { revert_round(initial, threat); }
    vector_free(&defenses);
    return result;
}

void find_win_sequence(threat_tree_node_t* node, vector_t* point_sequence) {
    if (node->threat.type >= PAT_A4) {
        vector_t threat_sequence = vector_new(threat_info_t, NULL);
        record_sequence(node, &threat_sequence);
        if (verify_sequence(threat_sequence)) {
            // print_sequence(threat_sequence);
            // prompt_pause();
            record_point_sequence(node, point_sequence);
            return;
        }
        vector_free(&threat_sequence);
    }
    for (threat_tree_node_t* child = node->son; child; child = child->brother) {
        find_win_sequence(child, point_sequence);
        if (point_sequence->size) return;
    }
}

static bool same_line(point_t pos1, point_t pos2) {
    return pos1.x == pos2.x || pos1.y == pos2.y || pos1.x - pos1.y == pos2.x - pos2.y ||
           pos1.x + pos1.y == pos2.x + pos2.y;
}

static int distance(point_t pos1, point_t pos2) {
    return max(abs(pos1.x - pos2.x), abs(pos1.y - pos2.y));
}

static threat_tree_node_t* merge_node;

void merge(threat_tree_node_t* node) {
    if (!merge_node) return;
    // log("merge: %c%d <-> %c%d", READABLE_POS(node->threat.action),
    // READABLE_POS(merge_node->threat.action));
    point_t p1 = node->threat.action, p2 = merge_node->threat.action;
    if (same_line(p1, p2) && distance(p1, p2) < 5) {
        // log("check compability");
        if (compatible(node, merge_node)) {
            // log("add!");
            add_threat(node, merge_node->threat);
        }
    }
    // log("done.");
}

void for_each_node(int depth, threat_tree_node_t* root, void (*func)(threat_tree_node_t*)) {
    if (!root) return;
    if (!depth) return;
    func(root);
    for (threat_tree_node_t* child = root->son; child; child = child->brother) {
        for_each_node(depth - 1, child, func);
    }
}

void combine_forest(vector_t forest) {
    for (size_t i = 0; i < forest.size; i++) {
        for (size_t j = 0; j < forest.size; j++) {
            if (i == j) continue;
            threat_tree_node_t* root1 = vector_get(threat_tree_node_t*, forest, i);
            threat_tree_node_t* root2 = vector_get(threat_tree_node_t*, forest, j);
            merge_node = root2;
            for_each_node(DEPTH_LIMIT, root1, merge);
        }
    }
}

const static int time_limits[] = {200, 1000, 1000};

vector_t get_threat_forest(board_t board, int id, bool only_four) {
    offend_id = id, defend_id = 3 - id;
    memcpy(initial, board, sizeof(board_t));
    vector_t threats = scan_threats_info(board, id, only_four);
    start_time = record_time(), TIME_LIMIT = time_limits[0], DEPTH_LIMIT = 12;
    vector_t threat_forest = vector_new(threat_tree_node_t*, free_threat_tree);
    threats.free_func = NULL;  // take ownership to tree
    for_each(threat_info_t, threats, threat) {
        threat_tree_node_t* root = malloc(sizeof(threat_tree_node_t));
        root->threat = threat;
        root->parent = root->son = root->brother = NULL;
        root->only_four = only_four;
        memcpy(root->board, board, sizeof(board_t));
        act_round(root->board, threat);
        build_threat_tree(root);
        vector_push_back(threat_forest, root);
    }
    start_time = record_time();
    TIME_LIMIT = time_limits[1];
    for (size_t i = 0; i < threat_forest.size; i++) {
        combine_forest(threat_forest);
        if (get_time(start_time) > TIME_LIMIT) break;
    }
    vector_free(&threats);
    return threat_forest;
}

vector_t vcf(board_t board, int id) {
    vector_t forest = get_threat_forest(board, id, true);
    vector_t win_sequence = vector_new(point_t, NULL);
    int tim = record_time();
    for_each_ptr(threat_tree_node_t*, forest, proot) {
        threat_tree_node_t* root = *proot;
        find_win_sequence(root, &win_sequence);
        if (win_sequence.size) break;
        if (get_time(tim) > time_limits[2]) break;
    }
    vector_free(&forest);
    return win_sequence;
}

void print_vcf(vector_t point_array) {
    char buffer[1024] = {0};
    snprintf(buffer, sizeof(buffer), "%c%d", READABLE_POS(vector_get(point_t, point_array, 0)));
    for (size_t i = 1; i < point_array.size; i++) {
        snprintf(buffer, sizeof(buffer), "%s -> %c%d", buffer,
                 READABLE_POS(vector_get(point_t, point_array, i)));
    }
    log("%s", buffer);
}