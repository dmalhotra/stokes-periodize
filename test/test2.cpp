// export OMP_NUM_THREADS=16; time make DEBUG=0 -B bin/test2 && time ./bin/test2 # not MPI capable yet
// Straight channel with a sphere inside, 3-periodic and 1-periodic formulations.

#include "periodize.hpp"
#include "utils.hpp"

/**
 * Compute the weighted sum of vals with weights wts.
 * vals: [x1,y1,z1, x2,y2,z2, ..., xN,yN,zN]
 * wts:  [w1, w2, ..., wN]
 * I: [mx, my, mz]
 */
// TODO: add MPI support
template <class Real> void SurfaceIntegral(sctl::Vector<Real>& I, const sctl::Vector<Real>& vals, const sctl::Vector<Real>& wts) {
  const sctl::Long dof = vals.Dim() / wts.Dim();
  SCTL_ASSERT(vals.Dim() == wts.Dim() * dof);
  if (I.Dim() != dof) I.ReInit(dof);
  I = 0;
  for (sctl::Long i = 0; i < wts.Dim(); i++) {
    for (sctl::Long j = 0; j < dof; j++) {
      I[j] += vals[i*dof + j] * wts[i];
    }
  }
}

/**
 * Add vector c0 to each point in vals.
 * vals: [x1,y1,z1, x2,y2,z2, ..., xN,yN,zN]
 * c0:   [cx,cy,cz]
 */
template <class Real> void AddConstVec(sctl::Vector<Real>& vals, const sctl::Vector<Real>& c0) {
  const sctl::Long dof = c0.Dim();
  const sctl::Long N = vals.Dim() / dof;
  SCTL_ASSERT(vals.Dim() == N * dof);
  for (sctl::Long i = 0; i < N; i++) {
    for (sctl::Long j = 0; j < dof; j++) {
      vals[i*dof + j] += c0[j];
    }
  }
}

/**
 * Background flow with unit pressure gradient along X-axis.
 */
template <class Real> sctl::Vector<Real> bg_flow(const sctl::Vector<Real>& X) {
  const sctl::Long N = X.Dim()/3;
  sctl::Vector<Real> U(N*3);
  for (sctl::Long i = 0; i < N; i++) {
    const auto x = X.begin() + i*3;
    U[i*3+0] = -((x[1]-0.5)*(x[1]-0.5) + (x[2]-0.5)*(x[2]-0.5))/4;
    U[i*3+1] = 0;
    U[i*3+2] = 0;
  }
  return U;
}

/**
 * Build element list for a straight channel with a sphere inside.
 */
template <class Real> sctl::SlenderElemList<Real> build_elem_lst(const sctl::Long Nelem_channel, const sctl::Long ElemOrder, const sctl::Long FourierOrder, sctl::Vector<Real>* NormalOrient_ptr = nullptr) {
  sctl::Vector<Real> Xc, eps, orient;
  sctl::Vector<sctl::Long> ElemOrderVec, FourierOrderVec;
  for (sctl::Long i = 0; i < Nelem_channel; i++) {
    ElemOrderVec.PushBack(ElemOrder);
    FourierOrderVec.PushBack(FourierOrder);
    const sctl::Vector<Real>& nodes = sctl::SlenderElemList<Real>::CenterlineNodes(ElemOrderVec[i]);
    for (sctl::Long j = 0; j < ElemOrderVec[i]; j++) {
      const Real x = (i+nodes[j])/Nelem_channel;
      Xc.PushBack(x);
      Xc.PushBack(0.5);
      Xc.PushBack(0.5);
      eps.PushBack(0.2);

      orient.PushBack(0);
      orient.PushBack(0);
      orient.PushBack(1);
    }
  }

  const sctl::Long Nelem_sphere = 4;
  for (sctl::Long i = 0; i < Nelem_sphere; i++) { // add a sphere
    ElemOrderVec.PushBack(ElemOrder);
    FourierOrderVec.PushBack(FourierOrder);
    const sctl::Vector<Real>& nodes = sctl::SlenderElemList<Real>::CenterlineNodes(ElemOrderVec[i]);
    for (sctl::Long j = 0; j < ElemOrderVec[i]; j++) {
      const Real r = 0.1;
      const Real theta = sctl::const_pi<Real>() * (i+nodes[j])/Nelem_sphere;
      Xc.PushBack(0.5+r*sctl::cos<Real>(theta));
      Xc.PushBack(0.5);
      Xc.PushBack(0.5);
      eps.PushBack(r*sctl::sin<Real>(theta));

      orient.PushBack(0);
      orient.PushBack(0);
      orient.PushBack(1);
    }
  }

  sctl::SlenderElemList<Real> elem_lst0(ElemOrderVec, FourierOrderVec, Xc, eps, orient);

  if (NormalOrient_ptr != nullptr) {
    NormalOrient_ptr->ReInit(0);
    constexpr sctl::Integer COORD_DIM = 3;
    sctl::Vector<sctl::Long> elem_wise_node_cnt;
    elem_lst0.GetNodeCoord(nullptr, nullptr, &elem_wise_node_cnt);
    for (sctl::Long i = 0; i < elem_wise_node_cnt.Dim(); i++) {
      for (sctl::Long j = 0; j < elem_wise_node_cnt[i]*COORD_DIM; j++) {
        NormalOrient_ptr->PushBack(i < Nelem_channel ? -1 : 1);
      }
    }
  }

  return elem_lst0;
}

template <class Real> void test(const sctl::Comm& comm) {
  // Combine single-layer and double-layer kernels in these proportions
  const Real SL_scal = 1.0;
  const Real DL_scal = 1.0;

  const Real pressure_drop = -1.0;
  const Real period_length = 1;

  const Real tol = 1e-14;
  const Real gmres_tol = 1e-12;
  const sctl::Long gmres_max_iter = 200;
  const sctl::Long Nelem_channel = 8;
  const sctl::Long ElemOrder = 10;
  const sctl::Long FourierOrder = 28;

  sctl::Vector<Real> NormalOrient; // normal orientation (1 if normal into fluid, else -1)
  const auto elem_lst0 = build_elem_lst(Nelem_channel, ElemOrder, FourierOrder, &NormalOrient); // geometry in the unit box [0,1]^3

  Real surface_area;
  sctl::Vector<Real> X0, wts;
  elem_lst0.GetNodeCoord(&X0, nullptr, nullptr);
  { // get wts and surface area
    sctl::Vector<Real> X, Xn, dist_far, surface_area_;
    sctl::Vector<sctl::Long> element_wise_node_cnt;
    elem_lst0.GetFarFieldNodes(X, Xn, wts, dist_far, element_wise_node_cnt, 1);
    SurfaceIntegral(surface_area_, wts*0+1, wts);
    surface_area = surface_area_[0];
  }


  StokesBIO<Real> LayerPotenOp0(SL_scal, DL_scal, comm);
  LayerPotenOp0.AddElemList(elem_lst0);
  LayerPotenOp0.SetAccuracy(tol);

  // Define the boundary-integral operator: (I/2 + D + S)[sigma-sigma_mean] + sigma_mean
  const auto BIO = [&wts,&surface_area,&elem_lst0,&LayerPotenOp0,&DL_scal,&NormalOrient](sctl::Vector<Real>* U, const sctl::Vector<Real>& sigma) {
    sctl::Vector<Real> sigma_mean, sigma0;
    { // compute sigma_mean and sigma0 = sigma - sigma_mean
      sctl::Vector<Real> sigma_;
      elem_lst0.GetFarFieldDensity(sigma_, sigma);
      SurfaceIntegral(sigma_mean, sigma_, wts);
      sigma_mean *= (1/surface_area);

      sigma0 = sigma;
      AddConstVec(sigma0, -sigma_mean);
    }

    U->SetZero();
    LayerPotenOp0.ComputePotential(*U, sigma0);
    if (DL_scal && U->Dim() == sigma.Dim()) (*U) += sigma0*0.5*NormalOrient * DL_scal; // for double-layer

    AddConstVec(*U, sigma_mean);
  };

  { // three-periodic
    LayerPotenOp0.SetTargetCoord(X0);
    LayerPotenOp0.SetPeriodicity(sctl::Periodicity::XYZ, period_length);

    const auto eval_rhs = [&LayerPotenOp0,surface_area,period_length](const Real pressure_drop) { // BIOpSL( -pressure_drop * cross_sectional_area / surface_area )
      sctl::Vector<Real> force_density(LayerPotenOp0.Dim(0)); force_density = 0;
      AddConstVec(force_density, sctl::Vector<Real>{-pressure_drop * period_length*period_length / surface_area, 0, 0});

      sctl::Vector<Real> U0;
      LayerPotenOp0.ComputeSL(U0, force_density);
      return U0;
    };

    sctl::Vector<Real> sigma;
    sctl::GMRES<Real> solver(comm);
    solver(&sigma, BIO, eval_rhs(pressure_drop), gmres_tol, gmres_max_iter);
    elem_lst0.WriteVTK("vis/sigma-3p", sigma, comm);

    { // Evaluate in interior, and write visualization
      VolumeVis<Real> vol_vis(elem_lst0, comm);
      const auto& X0 = vol_vis.GetCoord(); // set new target coordinates
      LayerPotenOp0.SetTargetCoord(X0);
      sctl::Vector<Real> U;
      BIO(&U, sigma);
      U -= eval_rhs(pressure_drop);
      vol_vis.WriteVTK("vis/U-3p", U);
    }
  }

  { // one-periodic
    LayerPotenOp0.SetTargetCoord(X0);
    LayerPotenOp0.SetPeriodicity(sctl::Periodicity::X, period_length);

    sctl::Vector<Real> sigma;
    sctl::GMRES<Real> solver(comm);
    solver(&sigma, BIO, bg_flow(X0) * (pressure_drop/period_length), gmres_tol, gmres_max_iter);
    elem_lst0.WriteVTK("vis/sigma-1p", sigma, comm);

    { // Evaluate in interior, and write visualization
      VolumeVis<Real> vol_vis(elem_lst0, comm);
      const auto& X0 = vol_vis.GetCoord(); // set new target coordinates
      LayerPotenOp0.SetTargetCoord(X0);
      sctl::Vector<Real> U;
      BIO(&U, sigma);
      U -= bg_flow(X0) * (pressure_drop/period_length);
      vol_vis.WriteVTK("vis/U-1p", U);
    }
  }
}

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  using Real = double;

  {
    sctl::Profile::Enable(true);
    const sctl::Comm comm = sctl::Comm::World();
    test<Real>(comm);
  }

  sctl::Comm::MPI_Finalize();
  return 0;
}
