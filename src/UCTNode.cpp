/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "config.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "UCTNode.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GTP.h"
#include "GameState.h"
#include "Network.h"
#include "Utils.h"

using namespace Utils;

UCTNode::UCTNode(int vertex, float score) : m_move(vertex), m_score(score) {
}

bool UCTNode::first_visit() const {
    return m_visits == 0;
}

SMP::Mutex& UCTNode::get_mutex() {
    return m_nodemutex;
}

bool UCTNode::create_children(std::atomic<int>& nodecount,
                              GameState& state,
                              float& blackeval,
                              float& whiteeval,
                              float& raw_wr,
                              float min_psa_ratio,
                              int symmetry) {
    // check whether somebody beat us to it (atomic)
    if (!expandable(min_psa_ratio)) {
        return false;
    }
    // acquire the lock
    LOCK(get_mutex(), lock);
    // no successors in final state
    if (state.get_passes() >= 2) {
        return false;
    }
    // check whether somebody beat us to it (after taking the lock)
    if (!expandable(min_psa_ratio)) {
        return false;
    }
    // Someone else is running the expansion
    if (m_is_expanding) {
        return false;
    }
    // We'll be the one queueing this node for expansion, stop others
    m_is_expanding = true;
    lock.unlock();

    Network::Netresult raw_netlist;
    if (symmetry == -1) {
        raw_netlist = Network::get_scored_moves(&state, Network::Ensemble::RANDOM_SYMMETRY);
    }
    else {
        raw_netlist = Network::get_scored_moves(&state, Network::Ensemble::DIRECT, symmetry, true);
    }

    blackeval = m_net_blackeval = raw_netlist.black_winrate;
    whiteeval = m_net_whiteeval = raw_netlist.white_winrate;
    raw_wr = raw_netlist.raw_winrate;
    update(blackeval, whiteeval);

    std::vector<Network::ScoreVertexPair> nodelist;

    const auto to_move = state.board.get_to_move();
    auto legal_sum = 0.0f;
    for (auto i = 0; i < BOARD_SQUARES; i++) {
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state.board.get_vertex(x, y);
        if (state.is_move_legal(to_move, vertex)) {
            nodelist.emplace_back(raw_netlist.policy[i], vertex);
            legal_sum += raw_netlist.policy[i];
        }
    }
    nodelist.emplace_back(raw_netlist.policy_pass, FastBoard::PASS);
    legal_sum += raw_netlist.policy_pass;

    if (legal_sum > std::numeric_limits<float>::min()) {
        // re-normalize after removing illegal moves.
        for (auto& node : nodelist) {
            node.first /= legal_sum;
        }
    } else {
        // This can happen with new randomized nets.
        auto uniform_prob = 1.0f / nodelist.size();
        for (auto& node : nodelist) {
            node.first = uniform_prob;
        }
    }

    link_nodelist(nodecount, nodelist, min_psa_ratio);
    return true;
}

void UCTNode::link_nodelist(std::atomic<int>& nodecount,
                            std::vector<Network::ScoreVertexPair>& nodelist,
                            float min_psa_ratio) {
    assert(min_psa_ratio < m_min_psa_ratio_children);

    if (nodelist.empty()) {
        return;
    }

    // Use best to worst order, so highest go first
    std::stable_sort(rbegin(nodelist), rend(nodelist));

    LOCK(get_mutex(), lock);

    const auto max_psa = nodelist[0].first;
    const auto old_min_psa = max_psa * m_min_psa_ratio_children;
    const auto new_min_psa = max_psa * min_psa_ratio;
    if (new_min_psa > 0.0f) {
        m_children.reserve(
            std::count_if(cbegin(nodelist), cend(nodelist),
                [=](const auto& node) { return node.first >= new_min_psa; }
            )
        );
    } else {
        m_children.reserve(nodelist.size());
    }

    auto skipped_children = false;
    for (const auto& node : nodelist) {
        if (node.first < new_min_psa) {
            skipped_children = true;
        } else if (node.first < old_min_psa) {
            m_children.emplace_back(node.second, node.first);
            ++nodecount;
        }
    }

    m_min_psa_ratio_children = skipped_children ? min_psa_ratio : 0.0f;
    m_is_expanding = false;
}

const std::vector<UCTNodePointer>& UCTNode::get_children() const {
    return m_children;
}


int UCTNode::get_move() const {
    return m_move;
}

void UCTNode::virtual_loss() {
    m_virtual_loss += VIRTUAL_LOSS_COUNT;
}

void UCTNode::virtual_loss_undo() {
    m_virtual_loss -= VIRTUAL_LOSS_COUNT;
}

void UCTNode::update(float blackeval, float whiteeval) {
    m_visits++;
    accumulate_eval(blackeval, whiteeval);
}

bool UCTNode::has_children() const {
    return m_min_psa_ratio_children <= 1.0f;
}

bool UCTNode::expandable(const float min_psa_ratio) const {
    return min_psa_ratio < m_min_psa_ratio_children;
}

float UCTNode::get_score() const {
    return m_score;
}

void UCTNode::set_score(float score) {
    m_score = score;
}

int UCTNode::get_visits() const {
    return m_visits;
}

// Return the true score, without taking into account virtual losses.
float UCTNode::get_pure_eval(int tomove) const {
    auto visits = get_visits();
    assert(visits > 0);
    if (tomove == FastBoard::BLACK) {
        auto blackeval = get_blackevals();
        return static_cast<float>(blackeval / double(visits));
    }
    if (tomove == FastBoard::WHITE) {
        auto whiteeval = get_whiteevals();
        return static_cast<float>(whiteeval / double(visits));
    }
}

float UCTNode::get_eval(int tomove, bool noflip) const {
    // Due to the use of atomic updates and virtual losses, it is
    // possible for the visit count to change underneath us. Make sure
    // to return a consistent result to the caller by caching the values.
    auto virtual_loss = int{m_virtual_loss};
    auto visits = get_visits() + virtual_loss;
    assert(visits > 0);
    auto blackeval = get_blackevals();
    auto whiteeval = get_whiteevals();
    if (noflip) {
        if (tomove == FastBoard::BLACK) {
            return static_cast<float>(blackeval / double(visits));
        } else {
            return static_cast<float>(whiteeval / double(visits));
        }
    } else {
        if (tomove == FastBoard::BLACK) {
            return static_cast<float>(-(whiteeval + static_cast<double>(virtual_loss)) / double(visits));
        } else {
            return static_cast<float>(-(blackeval + static_cast<double>(virtual_loss)) / double(visits));
        }
    }
}

float UCTNode::get_net_eval(int tomove) const {
    if (tomove == FastBoard::BLACK) {
        return m_net_blackeval;
    }
    if (tomove == FastBoard::WHITE) {
        return m_net_whiteeval;
    }
}

double UCTNode::get_blackevals() const {
    return m_blackevals;
}

double UCTNode::get_whiteevals() const {
    return m_whiteevals;
}

void UCTNode::accumulate_eval(float blackeval, float whiteeval) {
    atomic_add(m_blackevals, double(blackeval));
    atomic_add(m_whiteevals, double(whiteeval));
}

UCTNode* UCTNode::uct_select_child(int color, bool is_root) {
    LOCK(get_mutex(), lock);

    // Count parentvisits manually to avoid issues with transpositions.
    auto total_visited_policy = 0.0f;
    auto parentvisits = size_t{0};
    for (const auto& child : m_children) {
        if (child.valid()) {
            parentvisits += child.get_visits();
            if (child.get_visits() > 0) {
                total_visited_policy += child.get_score();
            }
        }
    }

    auto numerator = std::sqrt(double(parentvisits));
    auto fpu_reduction = 0.0f;
    // Lower the expected eval for moves that are likely not the best.
    // Do not do this if we have introduced noise at this node exactly
    // to explore more.
    
    auto parent_eval = get_pure_eval(color);
    auto opp_parent_eval = get_pure_eval(1 - color);
    if (!is_root || !cfg_noise) {
        fpu_reduction = cfg_fpu_reduction * std::sqrt(total_visited_policy);
        if (cfg_puct_factor == 2) {
            if (parent_eval < 0.5) {
                fpu_reduction *= parent_eval * (1 - parent_eval) / 0.25;
            } else {
                fpu_reduction *= opp_parent_eval * (1 - opp_parent_eval) / 0.25;
            }
        } else if (cfg_puct_factor == 1 || (cfg_puct_factor == 3 && parent_eval < 0.5)) {
            fpu_reduction *= parent_eval / 0.5;
        }
    }
    // Estimated eval for unknown nodes = current parent winrate - reduction
    float fpu_eval;
    if (parent_eval < 0.5) {
        fpu_eval = parent_eval - fpu_reduction;
    } else {
        fpu_eval = - opp_parent_eval - fpu_reduction;
    }

    auto best = static_cast<UCTNodePointer*>(nullptr);
    auto best_value = std::numeric_limits<double>::lowest();

    for (auto& child : m_children) {
        if (!child.active()) {
            continue;
        }

        auto winrate = fpu_eval;
        if (child.get_visits() > 0) {
            winrate = child.get_eval(color, parent_eval < 0.5);
        }
        auto psa = child.get_score();
        auto denom = 1.0 + child.get_visits();
        auto puct = cfg_puct * psa * (numerator / denom);
        if (cfg_puct_factor == 2) {
            if (parent_eval < 0.5) {
                puct *= parent_eval * (1 - parent_eval) / 0.25;
            } else {
                puct *= opp_parent_eval * (1 - opp_parent_eval) / 0.25;
            }
        } else if (cfg_puct_factor == 1 || (cfg_puct_factor == 3 && parent_eval < 0.5)) {
            puct *= parent_eval / 0.5;
        }
        auto value = winrate + puct;
        assert(value > std::numeric_limits<double>::lowest());

        if (value > best_value) {
            best_value = value;
            best = &child;
        }
    }

    assert(best != nullptr);
    best->inflate();
    return best->get();
}

class NodeComp : public std::binary_function<UCTNodePointer&,
                                             UCTNodePointer&, bool> {
public:
    NodeComp(int color) : m_color(color) {};
    bool operator()(const UCTNodePointer& a,
                    const UCTNodePointer& b) {
        // if visits are not same, sort on visits
        if (a.get_visits() != b.get_visits()) {
            return a.get_visits() < b.get_visits();
        }

        // neither has visits, sort on prior score
        if (a.get_visits() == 0) {
            return a.get_score() < b.get_score();
        }

        // both have same non-zero number of visits
        return a.get_pure_eval(m_color) < b.get_pure_eval(m_color) || a.get_pure_eval(1-m_color) > b.get_pure_eval(1-m_color);
    }
private:
    int m_color;
};

void UCTNode::sort_children(int color) {
    LOCK(get_mutex(), lock);
    std::stable_sort(rbegin(m_children), rend(m_children), NodeComp(color));
}

UCTNode& UCTNode::get_best_root_child(int color) {
    LOCK(get_mutex(), lock);
    assert(!m_children.empty());

    auto ret = std::max_element(begin(m_children), end(m_children),
                                NodeComp(color));
    ret->inflate();
    return *(ret->get());
}

size_t UCTNode::count_nodes() const {
    auto nodecount = size_t{0};
    nodecount += m_children.size();
    for (auto& child : m_children) {
        if (child.get_visits() > 0) {
            nodecount += child->count_nodes();
        }
    }
    return nodecount;
}

void UCTNode::invalidate() {
    m_status = INVALID;
}

void UCTNode::set_active(const bool active) {
    if (valid()) {
        m_status = active ? ACTIVE : PRUNED;
    }
}

bool UCTNode::valid() const {
    return m_status != INVALID;
}

bool UCTNode::active() const {
    return m_status == ACTIVE;
}
