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

  const auto build_elem_lst = [](const sctl::Long Nelem_channel, const sctl::Long ElemOrder, const sctl::Long FourierOrder){
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

    sctl::SlenderElemList<Real> elem_lst(ElemOrderVec, FourierOrderVec, Xc, eps, orient);
    return elem_lst;
  };
  const auto elem_lst0 = build_elem_lst(Nelem_channel, ElemOrder, FourierOrder); // geometry in the unit box [0,1]^3

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

  StokesBIO LayerPotenOp0(SL_scal, DL_scal, comm); // potential from elem_lst to X0
  LayerPotenOp0.AddElemList(elem_lst0);
  LayerPotenOp0.SetTargetCoord(X0);
  LayerPotenOp0.SetAccuracy(tol);
  LayerPotenOp0.SetPeriodicity(sctl::Periodicity::X, 1.0);

  // periodized layer potential operator
  const auto BIO = [&DL_scal,&LayerPotenOp0,&X0,NormalOrient](sctl::Vector<Real>* U, const sctl::Vector<Real>& sigma) {
    U->SetZero();
    LayerPotenOp0.ComputePotential(*U, sigma);
    if (DL_scal && U->Dim() == sigma.Dim()) (*U) -= sigma*0.5*NormalOrient * DL_scal; // for double-layer
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

