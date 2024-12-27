#include "vct.h"

#include "board.h"
#include "eval.h"
#include "pattern.h"
#include "util.h"
#include "vector.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define INF 0x7f7f7f7f

typedef struct threat_tree_node_t {
    board_t board;
    threat_info_t threat;
    struct threat_tree_node_t *parent, *son, *brother;
    int depth, subtree_size;
    int win_count, win_depth;
    vector_t win_nodes;      // threat_tree_node_t*
    vector_t best_sequence;  // point_t
    bool only_four;
} threat_tree_node_t;

void free_threat_info(void* ptr) {
    if (!ptr) return;
    threat_info_t* threat_info = ptr;
    vector_free(threat_info->consists);
    vector_free(threat_info->defenses);
}

threat_info_t clone_threat_info(const threat_info_t* threat_info) {
    threat_info_t result;
    memcpy(&result, threat_info, sizeof(threat_info_t));
    result.consists = vector_clone(threat_info->consists);
    result.defenses = vector_clone(threat_info->defenses);
    return result;
}

/// @brief get threat info (consists, defenses) of {threat} on {board}
threat_info_t attach_threat_info(board_t board, threat_t threat) {
    threat_info_t info = {.type = threat.pattern,
                          .consists = {0},
                          .defenses = {0},
                          .action = threat.pos,
                          .dir = threat.dir,
                          .id = threat.id};
    board[threat.pos.x][threat.pos.y] = threat.id;
    info.consists = find_relative_points(CONSIST, board, threat.pos, threat.dir.x, threat.dir.y);
    info.defenses = find_relative_points(DEFENSE, board, threat.pos, threat.dir.x, threat.dir.y);
    board[threat.pos.x][threat.pos.y] = 0;
    return info;
}

/// @return vector<threat_info_t, free_threat_info>
vector_t scan_threats_info(board_t board, int id, bool only_four) {
    vector_t threat_info = vector_new(threat_info_t, free_threat_info);
    vector_t threats = vector_new(threat_t, NULL);
    threat_storage_t storage = {0};
    storage[PAT_WIN] = storage[PAT_A4] = storage[PAT_D4] = &threats;
    if (!only_four) {
        storage[PAT_A3] = &threats;
    }
    scan_threats(board, id, id, storage);
    for_each(threat_t, threats, threat) {
        threat_info_t info = attach_threat_info(board, threat);
        vector_push_back(threat_info, info);
    }
    vector_free(threats);
    return threat_info;
}

/// @brief find threats dependent to {pos} on {board}
/// @return vector<threat_t>
vector_t find_threats(board_t board, point_t pos, bool only_four) {
    int id = board[pos.x][pos.y];
    vector_t result = vector_new(threat_t, NULL);
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
            get_attack_columns(value, id == 1, columns, 5);
            for (int i = 0; i < 5; i++) {
                if (columns[i] != -1) {
                    point_t np = column_to_point(pos, dx, dy, columns[i]);
                    board[np.x][np.y] = id;
                    pattern_t real_pat = get_pattern(board, np, dx, dy, id);
                    board[np.x][np.y] = 0;
                    if (real_pat != pat) continue;
                    // if (real_pat != pat) {
                    //     log_e("real: %s, expected: %s", pattern_typename[real_pat],
                    //           pattern_typename[pat]);
                    //     print_segment(get_segment(board, np, dx, dy), id == 1);
                    //     exit(1);
                    // }
                    if (!is_forbidden(board, np, id, 3)) {
                        threat_t threat = {
                            .dir = {dx, dy},
                            .id = id,
                            .pattern = pat,
                            .pos = np,
                        };
                        vector_push_back(result, threat);
                    }
                }
            }
        }
    }
    return result;
}

/// @return vector<threat_info_t, free_threat_info>
vector_t find_threats_info(board_t board, point_t pos, bool only_four) {
    vector_t result = vector_new(threat_info_t, free_threat_info);
    vector_t threats = find_threats(board, pos, only_four);
    for_each(threat_t, threats, threat) {
        threat_info_t info = attach_threat_info(board, threat);
        vector_push_back(result, info);
    }
    vector_free(threats);
    return result;
}

void free_tree_node(threat_tree_node_t* node) {
    free_threat_info(&node->threat);
    vector_free(node->win_nodes);
    vector_free(node->best_sequence);
}

int delete_threat_tree(threat_tree_node_t* root) {
    if (!root) return 0;
    int cnt = 1;
    for (threat_tree_node_t *child = root->son, *next; child; child = next) {
        next = child->brother;
        cnt += delete_threat_tree(child);
    }
    free_tree_node(root);
    free(root);
    return cnt;
}

typedef struct {
    int node_cnt, indent;
    int current_depth, DEPTH_LIMIT;
    int attack_id, defend_id;
    board_t initial;
    double start_time;
    struct {
        double build, merge, validate;
    } time_limit;
    double step_time_limit;
    threat_tree_node_t* merge_node;
} local_var_t;  // local var for each instance

const static double TIMEOUT_LIMIT = 100;

static void initialize_node(local_var_t* assets, threat_tree_node_t* node, board_t board) {
    memset(node, 0, sizeof(threat_tree_node_t));
    node->parent = node->son = node->brother = NULL;
    node->depth = 1;
    node->subtree_size = 1;
    node->win_count = 0;
    node->win_depth = INF;
    node->win_nodes = vector_new(threat_tree_node_t*, NULL);
    node->best_sequence = vector_new(point_t, NULL);
    memcpy(node->board, board, sizeof(board_t));
    assets->node_cnt++;
}

void free_threat_tree(void* ptr) {
    threat_tree_node_t** pnode = ptr;
    threat_tree_node_t* node = *pnode;
    delete_threat_tree(node);
    *pnode = NULL;
}

void print_threat(const local_var_t* assets, threat_info_t threat) {
    char indent_str[256] = {0}, defend_str[256] = {0};
    for (int i = 0; i < assets->indent; i++) strcat(indent_str, "  ");
    vector_serialize(defend_str, 256, ", ", threat.defenses, point_serialize);
    point_t pos = threat.action;
    log_l("%s%c%d[%d] %s(%d, %d)%s: %s, defenses: [%s]", indent_str, READABLE_POS(pos), threat.id,
          DARK, threat.dir.x, threat.dir.y, RESET, pattern_typename[threat.type], defend_str);
}

void print_threat_tree_node(const local_var_t* assets, threat_tree_node_t* node) {
    char indent_str[256] = {0};
    for (int i = 0; i < assets->indent; i++) strcat(indent_str, "  ");
    point_t pos = node->threat.action;
    char buffer[1024] = {0};
    if (node->win_count) {
        snprintf(buffer, sizeof(buffer), ", depth = %d, count = %d", node->win_depth,
                 node->win_count);
    }
    log_l("%s%c%d[%d] %s(%d, %d)%s: %s%s", indent_str, READABLE_POS(pos), node->threat.id, DARK,
          node->threat.dir.x, node->threat.dir.y, RESET, pattern_typename[node->threat.type],
          buffer);
}

void print_threat_tree(local_var_t* assets, threat_tree_node_t* root) {
    if (root->depth) {
        print_threat_tree_node(assets, root);
        assets->indent++;
    } else {
        if (root->win_depth != INF) {
            log_l("(virtual node) depth = %d, count = %d", root->win_depth, root->win_count);
        } else {
            log_l("(virtual node)");
        }
    }
    for (threat_tree_node_t* child = root->son; child; child = child->brother) {
        print_threat_tree(assets, child);
    }
    if (root->depth) assets->indent--;
}

void act_round(board_t board, threat_info_t threat) {
    board[threat.action.x][threat.action.y] = threat.id;
    for_each(point_t, threat.defenses, defend) { board[defend.x][defend.y] = 3 - threat.id; }
}

void revert_round(board_t board, threat_info_t threat) {
    board[threat.action.x][threat.action.y] = 0;
    for_each(point_t, threat.defenses, defend) { board[defend.x][defend.y] = 0; }
}

void build_threat_tree(local_var_t* assets, threat_tree_node_t* node);

bool threat_confilct(board_t board, threat_info_t info) {
    if (board[info.action.x][info.action.y]) return true;
    if (is_forbidden(board, info.action, info.id, 3)) return true;
    for_each(point_t, info.defenses, defend) {
        if (board[defend.x][defend.y]) return true;
    }
    return false;
}

bool chain_conflict(board_t board, threat_tree_node_t* chain[], int length) {
    board_t tmp;
    memcpy(tmp, board, sizeof(board_t));
    for (int i = 0; i < length; i++) {
        if (threat_confilct(tmp, chain[i]->threat)) return true;
        act_round(tmp, chain[i]->threat);
    }
    return false;
}

void add_threat(local_var_t* assets, threat_tree_node_t* node, threat_info_t threat) {
    if (node->threat.type >= PAT_A4) return;
    for (threat_tree_node_t* child = node->son; child; child = child->brother) {
        if (point_equal(child->threat.action, threat.action) &&
            point_equal(child->threat.dir, threat.dir)) {
            return;
        }
    }
    if (threat_confilct(node->board, threat)) {
        return;
    }
    threat_tree_node_t* child = malloc(sizeof(threat_tree_node_t));
    initialize_node(assets, child, node->board);
    act_round(child->board, threat);
    child->threat = clone_threat_info(&threat);
    child->parent = node;
    child->brother = node->son;
    child->only_four = node->only_four;
    child->depth = node->depth + 1;
    node->son = child;
    build_threat_tree(assets, child);
    node->subtree_size += child->subtree_size;
}

void build_threat_tree(local_var_t* assets, threat_tree_node_t* node) {
    if (get_time(assets->start_time) > assets->step_time_limit) return;
    if (assets->current_depth >= assets->DEPTH_LIMIT) return;
    assets->current_depth += 1;
    vector_t threats = find_threats_info(node->board, node->threat.action, node->only_four);
    for_each(threat_info_t, threats, threat) { add_threat(assets, node, threat); }
    assets->current_depth -= 1;
    vector_free(threats);
}

/// @param sequence vector<threat_info_t, NULL>
void record_sequence(const threat_tree_node_t* node, vector_t* sequence) {
    if (node->parent) record_sequence(node->parent, sequence);
    vector_push_back(*sequence, node->threat);
}

/// @param sequence vector<point_t>
void record_point_sequence(const threat_tree_node_t* node, vector_t* sequence) {
    if (node->parent) record_point_sequence(node->parent, sequence);
    if (node->depth) vector_push_back(*sequence, node->threat.action);
}

/// @param sequence vector<threat_info_t>
void print_threat_sequence(local_var_t* assets, vector_t sequence) {
    for_each(threat_info_t, sequence, threat) {
        print_threat(assets, threat);
        act_round(assets->initial, threat);
    }
    print(assets->initial);
    for_each(threat_info_t, sequence, threat) { revert_round(assets->initial, threat); }
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

/// @brief check if {node2}'s threat can be inserted into {node1}'s threat tree
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
    vector_free(seq1), vector_free(seq2);
    vector_free(actions1), vector_free(actions2);
    vector_free(defense1), vector_free(defense2);
    return result;
}

/// @brief find win nodes in the threat tree and mark it
void find_win_nodes(threat_tree_node_t* root) {
    if (!root->son) {
        if (root->threat.type >= PAT_A4) {
            root->win_count = 1;
            vector_push_back(root->win_nodes, root);
        }
    } else {
        for (threat_tree_node_t* child = root->son; child; child = child->brother) {
            find_win_nodes(child);
        }
    }
}

/// @brief try to defend the chain from ( {start}, {leaf} ]
bool try_defend_chain(local_var_t* assets, board_t board, threat_tree_node_t* start,
                      threat_tree_node_t* leaf) {
    if (get_time(assets->start_time) > assets->step_time_limit) return true;
    if (start == leaf) return false;
    threat_tree_node_t *chain[leaf->depth - start->depth], *node = leaf;
    int length = leaf->depth - start->depth;
    for (int i = length - 1; i >= 0; i--) {
        chain[i] = node;
        node = node->parent;
    }
    // log_l("defend ");
    // print(board);
    // log_l("chain: ");
    // for (int i = 0; i < length; i++) {
    //     print_threat(assets, chain[i]->threat);
    // }
    if (chain_conflict(board, chain, length)) {
        // log_l("conflict!");
        return true;
    }
    vector_t five_defenses = vector_new(threat_t, NULL);
    vector_t alive_four_defenses = vector_new(threat_t, NULL);
    vector_t dead_four_defenses = vector_new(threat_t, NULL);
    threat_storage_t storage = {
        [PAT_WIN] = &five_defenses,
        [PAT_A4] = &alive_four_defenses,
        [PAT_D4] = &dead_four_defenses,
    };
    scan_threats(board, assets->defend_id, assets->defend_id, storage);
    // log_l("size: (5: %d, A4: %d, D4: %d)", five_defenses.size, alive_four_defenses.size,
    // dead_four_defenses.size);
    // for_each(threat_t, dead_four_defenses, d4) { print_threat(attach_threat_info(board, d4)); }
    bool result = false;
    if (five_defenses.size && chain[0]->threat.type < PAT_WIN) {
        for_each(threat_t, five_defenses, five) {
            if (point_equal(five.pos, chain[0]->threat.action)) continue;
            // log_l("five defense on %c%d", READABLE_POS(five.pos));
            result = true;
            goto ret;
        }
    }
    if (alive_four_defenses.size && chain[0]->threat.type < PAT_D4) {
        for_each(threat_t, alive_four_defenses, alive_four) {
            if (point_equal(alive_four.pos, chain[0]->threat.action)) continue;
            // log_l("four defense on %c%d", READABLE_POS(alive_four.pos));
            result = true;
            goto ret;
        }
    }
    if (dead_four_defenses.size && chain[0]->threat.type < PAT_D4) {
        for_each(threat_t, dead_four_defenses, dead_four) {
            if (point_equal(dead_four.pos, chain[0]->threat.action)) continue;
            threat_info_t info = attach_threat_info(board, dead_four);
            act_round(board, info);
            assets->indent++;
            // log_l("===>");
            bool defended = try_defend_chain(assets, board, start, leaf);
            // log_l("<===");
            assets->indent--;
            revert_round(board, info);
            free_threat_info(&info);
            if (defended) {
                result = true;
                goto ret;
            }
        }
    }
ret:
    vector_free(five_defenses), vector_free(alive_four_defenses), vector_free(dead_four_defenses);
    // log_l("result: %s", result ? "success" : "failed");
    return result;
}

/// @brief merge the win chain from {child} to {root}
void merge_win_chain(threat_tree_node_t* root, threat_tree_node_t* child) {
    root->win_count += child->win_count;
    int depth_diff = child->depth - root->depth;
    if (root->win_depth > child->win_depth + depth_diff) {
        root->win_depth = child->win_depth + depth_diff;
        root->best_sequence.size = 0;
        for (threat_tree_node_t* node = child; node != root; node = node->parent) {
            vector_push_back(root->best_sequence, node->threat.action);
        }
        vector_reverse(root->best_sequence);
        vector_cat(root->best_sequence, child->best_sequence);
    }
    vector_cat(root->win_nodes, child->win_nodes);
}

/// @brief validate the win nodes in the threat tree
void validate_win_nodes(local_var_t* assets, threat_tree_node_t* root) {
    if (!root->son) {
        if (root->win_count) {
            root->win_depth = root->threat.type == PAT_WIN ? 1 : 2;  // 5 or a4
        }
        return;
    }
    if (get_time(assets->start_time) > assets->step_time_limit) return;
    for (threat_tree_node_t* child = root->son; child; child = child->brother) {
        validate_win_nodes(assets, child);
    }
    vector_t five_defenses = vector_new(threat_t, NULL);
    vector_t alive_four_defenses = vector_new(threat_t, NULL);
    vector_t dead_four_defenses = vector_new(threat_t, NULL);
    threat_storage_t storage = {
        [PAT_WIN] = &five_defenses,
        [PAT_A4] = &alive_four_defenses,
        [PAT_D4] = &dead_four_defenses,
    };
    scan_threats(root->board, assets->defend_id, assets->defend_id, storage);
    for (threat_tree_node_t* child = root->son; child; child = child->brother) {
        //        print(root->board);
        //        vector_t points = vector_new(point_t, NULL);
        //        record_point_sequence(child, &points);
        //        print_points(points, PROMPT_LOG, " -> ");
        //        vector_free(points);
        //        log_l("%c%d, type: %s", READABLE_POS(child->threat.action),
        //        pattern_typename[child->threat.type]);
        if (!child->win_count) continue;
        switch (child->threat.type) {
            case PAT_WIN: merge_win_chain(root, child); break;
            case PAT_A4:
                if (five_defenses.size) {
                    bool defended = false;
                    for_each(threat_t, five_defenses, five) {
                        if (!point_equal(five.pos, child->threat.action)) {
                            defended = true;
                            break;
                        }
                    }
                    if (defended) continue;
                }
                merge_win_chain(root, child);
                break;
            case PAT_D4:
                if (five_defenses.size) {
                    bool defended = false;
                    for_each(threat_t, five_defenses, five) {
                        if (!point_equal(five.pos, child->threat.action)) {
                            defended = true;
                            break;
                        }
                    }
                    if (defended) continue;
                }
                merge_win_chain(root, child);
                break;
            case PAT_A3:
                if (five_defenses.size) continue;
                if (alive_four_defenses.size) continue;
                if (dead_four_defenses.size == 0) {
                    merge_win_chain(root, child);
                } else {
                    for_each_ptr(threat_tree_node_t*, child->win_nodes, ptr) {
                        bool exists_defend = false;
                        threat_tree_node_t* win_node = *ptr;
                        for_each(threat_t, dead_four_defenses, dead_four) {
                            threat_info_t info = attach_threat_info(root->board, dead_four);
                            act_round(root->board, info);
                            bool defended = try_defend_chain(assets, root->board, root, win_node);
                            // log_l("\n");
                            if (defended) {
                                exists_defend = true;
                            }
                            revert_round(root->board, info);
                            free_threat_info(&info);
                            if (exists_defend) break;
                        }
                        if (!exists_defend) {
                            merge_win_chain(root, win_node);
                        }
                    }
                }
                break;
            default:
                log_e("unexpected pattern type: %s", pattern_typename[child->threat.type]);
                break;
        }
    }
#undef merge
    vector_free(five_defenses), vector_free(alive_four_defenses), vector_free(dead_four_defenses);
}

static bool same_line(point_t pos1, point_t pos2) {
    return pos1.x == pos2.x || pos1.y == pos2.y || pos1.x - pos1.y == pos2.x - pos2.y ||
           pos1.x + pos1.y == pos2.x + pos2.y;
}

static int distance(point_t pos1, point_t pos2) {
    return max(abs(pos1.x - pos2.x), abs(pos1.y - pos2.y));
}

void merge_tree(local_var_t* assets, threat_tree_node_t* node) {
    if (!assets->merge_node) return;
    // log_l("merge: %c%d <-> %c%d", READABLE_POS(node->threat.action),
    // READABLE_POS(merge_node->threat.action));
    point_t p1 = node->threat.action, p2 = assets->merge_node->threat.action;
    if (same_line(p1, p2)) {
        int dist = distance(p1, p2);
        bool adjacent = assets->attack_id == 1 ? dist < WIN_LENGTH : dist < SEGMENT_LEN;
        // log_l("check compability");
        if (adjacent && compatible(node, assets->merge_node)) {
            // log_l("add!");
            add_threat(assets, node, assets->merge_node->threat);
        }
    }
    // log_l("done.");
}

void for_each_node(local_var_t* assets, int depth, threat_tree_node_t* root,
                   void (*func)(local_var_t* assets, threat_tree_node_t*)) {
    if (!root) return;
    if (!depth) return;
    func(assets, root);
    for (threat_tree_node_t* child = root->son; child; child = child->brother) {
        for_each_node(assets, depth - 1, child, func);
    }
}

/// @brief try to add the root threat of one tree to another tree
void combine_forest(local_var_t* assets, vector_t forest) {
    for (size_t i = 0; i < forest.size; i++) {
        for (size_t j = 0; j < forest.size; j++) {
            if (i == j) continue;
            threat_tree_node_t* root1 = vector_get(threat_tree_node_t*, forest, i);
            threat_tree_node_t* root2 = vector_get(threat_tree_node_t*, forest, j);
            assets->merge_node = root2;
            for_each_node(assets, assets->DEPTH_LIMIT, root1, merge_tree);
        }
    }
}

/// @brief get threat forest of a board, merged it to a virtual node (no threat contained)
threat_tree_node_t* get_threat_tree(local_var_t* assets, board_t board, vector_t threats, int id,
                                    bool only_four) {
    assets->attack_id = id, assets->defend_id = 3 - id;
    memcpy(assets->initial, board, sizeof(board_t));
    vector_t threat_forest =
        vector_new(threat_tree_node_t*, NULL);  // give ownership to the super_root
    for_each(threat_info_t, threats, threat) {
        // print_threat(assets, threat);
        threat_tree_node_t* root = malloc(sizeof(threat_tree_node_t));
        initialize_node(assets, root, board);
        root->threat = clone_threat_info(&threat);
        root->only_four = only_four;
        act_round(root->board, threat);
        build_threat_tree(assets, root);
        vector_push_back(threat_forest, root);
    }
    for (size_t i = 0; i < threat_forest.size; i++) {
        combine_forest(assets, threat_forest);
        if (get_time(assets->start_time) > assets->step_time_limit) break;
    }
    threat_tree_node_t* super_root = malloc(sizeof(threat_tree_node_t));
    initialize_node(assets, super_root, board);
    super_root->depth = 0;  // set it to virtual node (do not contain any threat)
    for_each_ptr(threat_tree_node_t*, threat_forest, proot) {
        threat_tree_node_t* root = *proot;
        root->brother = super_root->son;
        root->parent = super_root;
        super_root->son = root;
        super_root->subtree_size += root->subtree_size;
    }
    vector_free(threat_forest);
    return super_root;
}

int max_vct_depth = 0;

/// @brief find Victory by Continuous Threats sequence
/// @return vector<point_t>
vector_t vct(bool only_four, board_t board, int id, double time_ms) {
    vector_t sequence = vector_new(point_t, NULL);
    double tim = record_time();
    local_var_t assets = {0};
    assets.step_time_limit = min(time_ms * 0.4, TIMEOUT_LIMIT);
    int depth;
    vector_t threats = scan_threats_info(board, id, only_four);
    for (depth = 2; depth < 30; depth++) {
        if (get_time(tim) > time_ms) break;
        assets.node_cnt = 0;
        assets.DEPTH_LIMIT = depth;
        assets.start_time = record_time();
        threat_tree_node_t* root = get_threat_tree(&assets, board, threats, id, only_four);
        find_win_nodes(root);
        assets.start_time = record_time();
        validate_win_nodes(&assets, root);
        // log_l("depth %d, size: %d", depth, root->subtree_size);
        if (root->win_depth != INF) {
            vector_copy(sequence, root->best_sequence);
            delete_threat_tree(root);
            max_vct_depth = max(max_vct_depth, depth);
            break;
        } else {
            delete_threat_tree(root);
        }
    }
    vector_free(threats);
    return sequence;
}

vector_t complex_vct(bool only_four, board_t board, int self_id, double time_ms, int depth) {
    const int oppo_id = 3 - self_id;
    vector_t sequence = vct(only_four, board, self_id, time_ms / 2);
    if (sequence.size || !depth) return sequence;
    vector_t attacks = scan_threats_info(board, self_id, false);
    vector_t critical_defends = scan_threats_by_threshold(board, oppo_id, PAT_A4);
    vector_t defends = vector_new(threat_t, NULL);
    scan_threats(board, oppo_id, oppo_id, (threat_storage_t){[PAT_D4] = &defends});
#define free_threats()    \
    vector_free(attacks); \
    vector_free(critical_defends)
    if (critical_defends.size) {
        free_threats();
        return sequence;
    }
    int situation_count = 0;
    for_each(threat_info_t, attacks, attack) { situation_count += attack.defenses.size; }
    for_each(threat_info_t, attacks, attack) {
        bool exists_defend = false;
        vector_t cur_seq = {0};
        vector_t defensive_points = vector_new(point_t, NULL);
        vector_cat(defensive_points, attack.defenses);
        if (attack.type == PAT_A3) {
            for_each(threat_t, defends, defend) { vector_push_back(defensive_points, defend.pos); }
        }
        for_each(point_t, defensive_points, defend_pos) {
            board[attack.action.x][attack.action.y] = self_id;
            board[defend_pos.x][defend_pos.y] = oppo_id;
            vector_free(cur_seq);
            cur_seq =
                complex_vct(only_four, board, self_id, (time_ms / 2) / situation_count, depth - 1);
            board[attack.action.x][attack.action.y] = 0;
            board[defend_pos.x][defend_pos.y] = 0;
            if (!cur_seq.size) {
                exists_defend = true;
                break;
            }
        }
        vector_free(defensive_points);
        if (!exists_defend) {
            vector_push_back(sequence, attack.action);
            vector_cat(sequence, cur_seq);
            break;
        }
    }
    free_threats();
    return sequence;
}