// export OMP_NUM_THREADS=16; time make DEBUG=0 -B bin/test1 && time mpirun -n 1 --map-by slot:pe=$OMP_NUM_THREADS ./bin/test1

#include "periodize.hpp"
#include "utils.hpp"

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

template <class Real> void test(sctl::Comm comm) {
  // Combine single-layer and double-layer kernels in these proportions
  const Real SL_scal = 1.0;
  const Real DL_scal = 1.0;

  const Real tol = 1e-14;
  const Real gmres_tol = 1e-11;
  const sctl::Long gmres_max_iter = 50;
  const sctl::Long Nelem_channel = 8;
  const sctl::Long ElemOrder = 10;
  const sctl::Long FourierOrder = 28;

  const auto build_elem_lst_nbr = [](const sctl::Long Nelem_channel, const sctl::Long ElemOrder, const sctl::Long FourierOrder, const bool add_nbrs){
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

    if (add_nbrs) { // duplicate geomtry to add images
      sctl::Vector<Real> Xc_, eps_, orient_;
      sctl::Vector<sctl::Long> ElemOrderVec_, FourierOrderVec_;
      for (sctl::Long k0 = -1; k0 <= 1; k0++) {
        for (const auto& x : eps) eps_.PushBack(x);
        for (const auto& x : orient) orient_.PushBack(x);
        for (const auto& x : ElemOrderVec) ElemOrderVec_.PushBack(x);
        for (const auto& x : FourierOrderVec) FourierOrderVec_.PushBack(x);
        for (sctl::Long i = 0; i < Xc.Dim()/3; i++) { // shift in x
          Xc_.PushBack(Xc[i*3+0] + k0);
          Xc_.PushBack(Xc[i*3+1]);
          Xc_.PushBack(Xc[i*3+2]);
        }
      }
      Xc.Swap(Xc_);
      eps.Swap(eps_);
      orient.Swap(orient_);
      ElemOrderVec.Swap(ElemOrderVec_);
      FourierOrderVec.Swap(FourierOrderVec_);
    }
    sctl::SlenderElemList<Real> elem_lst(ElemOrderVec, FourierOrderVec, Xc, eps, orient);
    return elem_lst;
  };
  const auto elem_lst0 = build_elem_lst_nbr(Nelem_channel, ElemOrder, FourierOrder, false); // geometry in the unit box [0,1]^3
  const auto elem_lst_nbr = build_elem_lst_nbr(Nelem_channel, ElemOrder, FourierOrder, true); // geometry with one set of images in each direction
  const sctl::Long Nrepeat = elem_lst_nbr.Size() / elem_lst0.Size(); // should be 3
  //elem_lst_nbr.WriteVTK("vis/S-nbr", comm);

  sctl::Vector<Real> X0; // target coordinates
  elem_lst0.GetNodeCoord(&X0, nullptr, nullptr);
  const auto X_proxy = Periodize<Real>::GetProxySurf(); // proxy points coordinates

  sctl::Vector<Real> NormalOrient; // normal orientation (interior=1, exterior=-1)
  { // set NormalOrient
    constexpr sctl::Integer COORD_DIM = 3;
    sctl::Vector<sctl::Long> elem_wise_node_cnt;
    elem_lst0.GetNodeCoord(nullptr, nullptr, &elem_wise_node_cnt);
    for (sctl::Long i = 0; i < elem_wise_node_cnt.Dim(); i++) {
      for (sctl::Long j = 0; j < elem_wise_node_cnt[i]*COORD_DIM; j++) {
        NormalOrient.PushBack(i < Nelem_channel ? 1 : -1);
      }
    }
  }

  StokesBIO LayerPotenOp0(SL_scal, DL_scal, comm); // potential from elem_lst_nbr to X0
  LayerPotenOp0.AddElemList(elem_lst_nbr);
  LayerPotenOp0.SetTargetCoord(X0);
  LayerPotenOp0.SetAccuracy(tol);

  StokesBIO LayerPotenOp_proxy(SL_scal, DL_scal, comm); // potential from elem_lst0 to proxy points
  LayerPotenOp_proxy.AddElemList(elem_lst0);
  LayerPotenOp_proxy.SetTargetCoord(X_proxy);
  LayerPotenOp_proxy.SetAccuracy(tol);

  // periodized layer potential operator
  const auto BIO = [&DL_scal,&LayerPotenOp0,&LayerPotenOp_proxy,&X0,&Nrepeat,NormalOrient](sctl::Vector<Real>* U, const sctl::Vector<Real>& sigma) {
    const sctl::Long N = sigma.Dim();

    sctl::Vector<Real> sigma_nbr(Nrepeat*N); // repeat sigma Nrepeat times
    for (sctl::Long k = 0; k < Nrepeat; k++) {
      for (sctl::Long i = 0; i < N; i++) {
        sigma_nbr[k*N+i] = sigma[i];
      }
    }

    U->SetZero();
    LayerPotenOp0.ComputePotential(*U, sigma_nbr);
    if (DL_scal && U->Dim() == N) (*U) -= sigma*0.5*NormalOrient * DL_scal; // for double-layer

    { // Add far-field
      sctl::Vector<Real> U_proxy, U_far;
      LayerPotenOp_proxy.ComputePotential(U_proxy, sigma);
      Periodize<Real>::EvalFarField(U_far, X0, U_proxy);
      (*U) += U_far;
    }
  };

  // Solve for sigma to satisfy no-slip boundary conditions: BIO(sigma) + bg_flow = 0
  sctl::Vector<Real> sigma;
  sctl::GMRES<Real> solver(comm);
  solver(&sigma, BIO, -bg_flow(X0), gmres_tol, gmres_max_iter);
  elem_lst0.WriteVTK("vis/sigma", sigma, comm);

  { // Evaluate in interior, and write visualization
    VolumeVis<Real> vol_vis(elem_lst0, comm);
    X0 = vol_vis.GetCoord(); // set new target coordinates
    LayerPotenOp0.SetTargetCoord(X0);
    sctl::Vector<Real> U;
    BIO(&U, sigma);
    U += bg_flow(X0);
    vol_vis.WriteVTK("vis/U", U);
  }
}

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  using Real = double;

  {
    //sctl::Profile::Enable(true);
    sctl::Comm comm = sctl::Comm::World();
    test<Real>(comm);
  }

  sctl::Comm::MPI_Finalize();
  return 0;
}

