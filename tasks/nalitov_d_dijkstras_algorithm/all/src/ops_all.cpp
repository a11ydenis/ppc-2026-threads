#include "nalitov_d_dijkstras_algorithm/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "nalitov_d_dijkstras_algorithm/common/include/common.hpp"

namespace nalitov_d_dijkstras_algorithm {

namespace {

bool CheckedSum(std::int64_t acc, Cost addend, std::int64_t &out) {
  const auto x = static_cast<std::int64_t>(addend);
  if (x > 0 && acc > std::numeric_limits<std::int64_t>::max() - x) {
    return false;
  }
  if (x < 0 && acc < std::numeric_limits<std::int64_t>::min() - x) {
    return false;
  }
  out = acc + x;
  return true;
}

void MinDistVertexOp(void *in_buf, void *inout_buf, const int *len, MPI_Datatype * /*dtype*/) {
  const auto *in = static_cast<const int *>(in_buf);
  auto *inout = static_cast<int *>(inout_buf);
  for (int i = 0; i < *len; i += 2) {
    const int a_cost = in[i];
    const int a_vtx = in[i + 1];
    const int b_cost = inout[i];
    const int b_vtx = inout[i + 1];
    const bool a_wins = (a_cost < b_cost) || (a_cost == b_cost && a_vtx != -1 && (b_vtx == -1 || a_vtx < b_vtx));
    if (a_wins) {
      inout[i] = a_cost;
      inout[i + 1] = a_vtx;
    }
  }
}

}  // namespace

NalitovDDijkstrasAlgorithmALL::NalitovDDijkstrasAlgorithmALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool NalitovDDijkstrasAlgorithmALL::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  if (rank_ != 0) {
    return true;
  }
  if (GetOutput() != 0) {
    return false;
  }
  const InType &in = GetInput();
  constexpr int kMaxVertices = 10000;
  if (in.n <= 0 || in.n > kMaxVertices) {
    return false;
  }
  if (in.source < 0 || in.source >= in.n) {
    return false;
  }
  const auto arc_ok = [&in](const Arc &a) {
    return a.from >= 0 && a.to >= 0 && a.from < in.n && a.to < in.n && a.weight >= 0;
  };
  return std::ranges::all_of(in.arcs, arc_ok);
}

bool NalitovDDijkstrasAlgorithmALL::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  std::array<int, 3> header{};
  if (rank_ == 0) {
    const InType &in = GetInput();
    header[0] = in.n;
    header[1] = in.source;
    header[2] = static_cast<int>(in.arcs.size());
  }
  MPI_Bcast(header.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
  n_ = header[0];
  source_ = header[1];
  const int arc_count = header[2];

  if (n_ <= 0) {
    return false;
  }

  std::vector<int> arc_buf(static_cast<std::size_t>(arc_count) * 3);
  if (rank_ == 0) {
    const InType &in = GetInput();
    for (int i = 0; i < arc_count; ++i) {
      const Arc &a = in.arcs[static_cast<std::size_t>(i)];
      arc_buf[static_cast<std::size_t>(i) * 3 + 0] = a.from;
      arc_buf[static_cast<std::size_t>(i) * 3 + 1] = a.to;
      arc_buf[static_cast<std::size_t>(i) * 3 + 2] = static_cast<int>(a.weight);
    }
  }
  if (arc_count > 0) {
    MPI_Bcast(arc_buf.data(), arc_count * 3, MPI_INT, 0, MPI_COMM_WORLD);
  }

  graph_.assign(static_cast<std::size_t>(n_), {});
  for (int i = 0; i < arc_count; ++i) {
    const int from = arc_buf[static_cast<std::size_t>(i) * 3 + 0];
    const int to = arc_buf[static_cast<std::size_t>(i) * 3 + 1];
    const int weight = arc_buf[static_cast<std::size_t>(i) * 3 + 2];
    graph_[static_cast<std::size_t>(from)].emplace_back(static_cast<NodeId>(to), static_cast<Cost>(weight));
  }

  dist_.assign(static_cast<std::size_t>(n_), kInf);
  visited_.assign(static_cast<std::size_t>(n_), 0);
  dist_[static_cast<std::size_t>(source_)] = 0;

  const int base = n_ / size_;
  const int rem = n_ % size_;
  local_start_ = rank_ * base + std::min(rank_, rem);
  local_count_ = base + (rank_ < rem ? 1 : 0);

  GetOutput() = 0;
  return true;
}

bool NalitovDDijkstrasAlgorithmALL::RunImpl() {
  if (static_cast<int>(graph_.size()) != n_) {
    return false;
  }

  MPI_Op min_dist_op = MPI_OP_NULL;
  MPI_Op_create(reinterpret_cast<MPI_User_function *>(
                    static_cast<void (*)(void *, void *, const int *, MPI_Datatype *)>(MinDistVertexOp)),
                /*commute=*/1, &min_dist_op);

  for (int step = 0; step < n_; ++step) {
    Cost proc_best_cost = kInf;
    NodeId proc_best_vtx = -1;

#pragma omp parallel default(none) shared(proc_best_cost, proc_best_vtx)
    {
      Cost thr_cost = kInf;
      NodeId thr_vtx = -1;

#pragma omp for nowait schedule(static)
      for (int vi = local_start_; vi < local_start_ + local_count_; ++vi) {
        if (visited_[static_cast<std::size_t>(vi)] != 0) {
          continue;
        }
        const Cost d = dist_[static_cast<std::size_t>(vi)];
        const bool better = d < thr_cost;
        const bool tie = (d == thr_cost) && (thr_vtx == -1 || vi < thr_vtx);
        if (better || tie) {
          thr_cost = d;
          thr_vtx = vi;
        }
      }

#pragma omp critical
      {
        const bool better = thr_cost < proc_best_cost;
        const bool tie =
            (thr_cost == proc_best_cost) && (thr_vtx != -1) && (proc_best_vtx == -1 || thr_vtx < proc_best_vtx);
        if (better || tie) {
          proc_best_cost = thr_cost;
          proc_best_vtx = thr_vtx;
        }
      }
    }

    std::array<int, 2> local_pair = {static_cast<int>(proc_best_cost), static_cast<int>(proc_best_vtx)};
    std::array<int, 2> global_pair = {static_cast<int>(kInf), -1};
    MPI_Allreduce(local_pair.data(), global_pair.data(), 2, MPI_INT, min_dist_op, MPI_COMM_WORLD);

    const NodeId pivot = static_cast<NodeId>(global_pair[1]);
    if (pivot == -1) {
      break;
    }

    visited_[static_cast<std::size_t>(pivot)] = 1;

    const Cost d_pivot = dist_[static_cast<std::size_t>(pivot)];
    if (d_pivot == kInf) {
      break;
    }

    const auto &nbrs = graph_[static_cast<std::size_t>(pivot)];
    const auto nbr_count = static_cast<int>(nbrs.size());

#pragma omp parallel for default(none) shared(nbrs, d_pivot) schedule(static)
    for (int ei = 0; ei < nbr_count; ++ei) {
      const NodeId tgt = nbrs[static_cast<std::size_t>(ei)].first;
      const Cost w = nbrs[static_cast<std::size_t>(ei)].second;
      if (visited_[static_cast<std::size_t>(tgt)] != 0) {
        continue;
      }
      if (d_pivot > kInf - w) {
        continue;
      }
      const Cost cand = d_pivot + w;
#pragma omp critical
      {
        if (cand < dist_[static_cast<std::size_t>(tgt)]) {
          dist_[static_cast<std::size_t>(tgt)] = cand;
        }
      }
    }
  }

  MPI_Op_free(&min_dist_op);

  std::int64_t local_sum = 0;
  for (int vi = local_start_; vi < local_start_ + local_count_; ++vi) {
    const Cost d = dist_[static_cast<std::size_t>(vi)];
    if (d == kInf) {
      continue;
    }
    std::int64_t tmp = 0;
    if (CheckedSum(local_sum, d, tmp)) {
      local_sum = tmp;
    }
  }

  std::int64_t global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    GetOutput() = global_sum;
  }
  return true;
}

bool NalitovDDijkstrasAlgorithmALL::PostProcessingImpl() {
  return rank_ != 0 || GetOutput() >= 0;
}

}  // namespace nalitov_d_dijkstras_algorithm
