#include "periodize.hpp"

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
 * Reference solution for checking error.
 */
template <class Real> sctl::Vector<Real> u_ref(const sctl::Vector<Real>& X) {
  const sctl::Long N = X.Dim()/3;
  sctl::Vector<Real> U(N*3);
  for (sctl::Long i = 0; i < N; i++) {
    const auto x = X.begin() + i*3;
    U[i*3+0] = 1e-2 - ((x[1]-0.4)*(x[1]-0.4) + (x[2]-0.3)*(x[2]-0.3))/4;
    U[i*3+1] = 0;
    U[i*3+2] = 0;
  }
  return U;
}

// Stokes combined field operator (S+D)
template <sctl::Long SL_scal> struct Stokes3D_CF_ {
  static const std::string& Name() {
    // Name determines what quadrature tables to use.
    // Single-layer quadrature tables also works for combined fields.
    static const std::string name = "Stokes3D-FxU";
    return name;
  }

  static constexpr sctl::Integer FLOPS() {
    return 50;
  }

  template <class Real> static constexpr Real uKerScaleFactor() {
    return 1 / (8 * sctl::const_pi<Real>());
  }

  template <sctl::Integer digits, class VecType>
  static void uKerMatrix(VecType (&u)[3][3], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
    using Real = typename VecType::ScalarType;
    const auto SL_scal_ = VecType((Real)SL_scal);
    const auto r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    const auto rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero()); // Compute inverse square root
    const auto rinv2 = rinv * rinv;
    const auto rinv3 = rinv2 * rinv;
    const auto rinv5 = rinv3 * rinv2;
    const auto rdotn = r[0] * n[0] + r[1] * n[1] + r[2] * n[2];
    const auto rdotn_rinv5_6 = VecType((Real)6) * rdotn * rinv5;
    for (sctl::Integer i = 0; i < 3; i++) {
      for (sctl::Integer j = 0; j < 3; j++) {
        const auto ri_rj = r[i] * r[j];
        const auto ker_dl = ri_rj * rdotn_rinv5_6; // Double-layer kernel
        const auto ker_sl = (i == j ? rinv + ri_rj * rinv3 : ri_rj * rinv3); // Single-layer kernel
        u[i][j] = ker_dl + ker_sl * SL_scal_; // Combine kernels
      }
    }
  }
};
using Stokes3D_CF = sctl::GenericKernel<Stokes3D_CF_<1>>;

/**
 * Visualize volume inside SlenderElemList.
 */
template <class Real> class VolumeVis {
    static constexpr sctl::Integer COORD_DIM = 3;
    static constexpr sctl::Integer s_order = 20;
    static constexpr sctl::Integer t_order = 60;
    static constexpr sctl::Integer r_order = 12;
  public:

    VolumeVis() = default;

    /**
     * @brief Construct a new VolumeVis object.
     *
     * @param elem_lst the geometry.
     * @param comm MPI communicator.
     */
    VolumeVis(const sctl::SlenderElemList<Real>& elem_lst, const sctl::Comm& comm = sctl::Comm::Self()) : comm_(comm) {
      Nelem = elem_lst.Size();
      sctl::Vector<Real> s_param, sin_theta, cos_theta;
      for (sctl::Long i = 0; i < s_order; i++) {
        const Real t = i/(Real)(s_order-1);
        s_param.PushBack(t);
      }
      for (sctl::Long i = 0; i < t_order; i++) {
        const Real t = i/(Real)t_order;
        sin_theta.PushBack(sctl::sin<Real>(2*sctl::const_pi<Real>()*t));
        cos_theta.PushBack(sctl::cos<Real>(2*sctl::const_pi<Real>()*t));
      }
      for (sctl::Long elem_idx = 0; elem_idx < Nelem; elem_idx++) {
        const Real t_order_inv = 1/(Real)t_order;
        const Real r_order_inv = (1-1e-6)/(Real)(r_order-1);
        sctl::Vector<Real> X_, Xc(COORD_DIM);
        elem_lst.GetGeom(&X_, nullptr, nullptr, nullptr, nullptr, s_param, sin_theta, cos_theta, elem_idx);
        for (sctl::Long i = 0; i < s_order; i++) {
          Xc = 0;
          for (sctl::Long j = 0; j < t_order; j++) {
            for (sctl::Long l = 0; l < COORD_DIM; l++) {
              Xc[l] += X_[(i*t_order+j)*COORD_DIM+l] * t_order_inv;
            }
          }
          for (sctl::Long j = 0; j < t_order; j++) {
            for (sctl::Long k = 0; k < r_order; k++) {
              for (sctl::Long l = 0; l < COORD_DIM; l++) {
                coord.PushBack((X_[(i*t_order+j)*COORD_DIM+l]-Xc[l])*k*r_order_inv + Xc[l]);
              }
            }
          }
        }
      }
    }

    /**
     * @brief Get the coordinates of the discretization points.
     *
     * @return const Vector<Real>& Vector containing the coordinates.
     */
    const sctl::Vector<Real>& GetCoord() const {
      return coord;
    }

    /**
     * @brief Write the volume to a VTK file.
     *
     * @param fname File name.
     * @param F Data associated with the discretization points.
     */
    void WriteVTK(const std::string& fname, const sctl::Vector<Real>& F) const {
      sctl::VTUData vtu_data;
      GetVTUData(vtu_data, F);
      vtu_data.WriteVTK(fname, comm_);
    }


    /**
     * @brief Get VTU data.
     *
     * @param vtu_data VTU data object.
     * @param F Data associated with the discretization points.
     */
    void GetVTUData(sctl::VTUData& vtu_data, const sctl::Vector<Real>& F) const {
      for (const auto& x : coord) vtu_data.coord.PushBack((float)x);
      for (const auto& x :     F) vtu_data.value.PushBack((float)x);
      for (sctl::Long l = 0; l < Nelem; l++) {
        const sctl::Long offset = l * s_order*t_order*r_order;
        for (sctl::Long i = 0; i < s_order-1; i++) {
          for (sctl::Long j = 0; j < t_order; j++) {
            for (sctl::Long k = 0; k < r_order-1; k++) {
              auto idx = [this,&offset](sctl::Long i, sctl::Long j, sctl::Long k) {
                return offset+(i*t_order+(j%t_order))*r_order+k;
              };
              vtu_data.connect.PushBack(idx(i+0,j+0,k+0));
              vtu_data.connect.PushBack(idx(i+0,j+0,k+1));
              vtu_data.connect.PushBack(idx(i+0,j+1,k+1));
              vtu_data.connect.PushBack(idx(i+0,j+1,k+0));
              vtu_data.connect.PushBack(idx(i+1,j+0,k+0));
              vtu_data.connect.PushBack(idx(i+1,j+0,k+1));
              vtu_data.connect.PushBack(idx(i+1,j+1,k+1));
              vtu_data.connect.PushBack(idx(i+1,j+1,k+0));
              vtu_data.offset.PushBack(vtu_data.connect.Dim());;
              vtu_data.types.PushBack(12);
            }
          }
        }
      }
    }

  private:

    sctl::Comm comm_;
    sctl::Long Nelem;
    sctl::Vector<Real> coord;
};

template <class Real> void test() {
  using KerType = Stokes3D_CF;
  //using KerType = sctl::Stokes3D_FxU; // unstable if gmres_tol is too small

  const Real tol = 1e-14;
  const Real gmres_tol = 1e-11;
  const sctl::Long Nelem = 4;
  const sctl::Long ElemOrder = 10;
  const sctl::Long FourierOrder = 28;

  const sctl::Comm comm = sctl::Comm::Self();
  const KerType stokes_ker;

  const auto build_elem_lst_nbr = [](const sctl::Long Nelem, const sctl::Long ElemOrder, const sctl::Long FourierOrder, const sctl::Integer nbr_range){
    sctl::Vector<Real> Xc, eps, orient;
    sctl::Vector<sctl::Long> ElemOrderVec, FourierOrderVec;
    for (sctl::Long k0 = -nbr_range; k0 <=nbr_range; k0++) {
      for (sctl::Long k1 = -nbr_range; k1 <=nbr_range; k1++) {
        for (sctl::Long k2 = -nbr_range; k2 <=nbr_range; k2++) {
          for (sctl::Long i = 0; i < Nelem; i++) {
            ElemOrderVec.PushBack(ElemOrder);
            FourierOrderVec.PushBack(FourierOrder);
            const sctl::Vector<Real>& nodes = sctl::SlenderElemList<Real>::CenterlineNodes(ElemOrderVec[i]);
            for (sctl::Long j = 0; j < ElemOrderVec[i]; j++) {
              const Real x = (i+nodes[j])/Nelem;
              Xc.PushBack(k0+x);
              Xc.PushBack(k1+0.4);
              Xc.PushBack(k2+0.3);
              eps.PushBack(0.2);

              orient.PushBack(0);
              orient.PushBack(0);
              orient.PushBack(1);
            }
          }
        }
      }
    }
    sctl::SlenderElemList<Real> elem_lst(ElemOrderVec, FourierOrderVec, Xc, eps, orient);
    return elem_lst;
  };
  const auto elem_lst0 = build_elem_lst_nbr(Nelem, ElemOrder, FourierOrder, 0); // geometry in the unit box [0,1]^3
  const auto elem_lst_nbr = build_elem_lst_nbr(Nelem, ElemOrder, FourierOrder, 1); // geometry with one set of images in each direction
  const sctl::Long Nrepeat = elem_lst_nbr.Size() / elem_lst0.Size(); // should be 3^3 = 27
  //elem_lst_nbr.WriteVTK("vis/S-nbr");

  sctl::Vector<Real> X0; // target coordinates
  elem_lst0.GetNodeCoord(&X0, nullptr, nullptr);
  const auto X_proxy = Periodize<Real>::GetProxySurf(); // proxy points coordinates

  sctl::BoundaryIntegralOp<Real, KerType> LayerPotenOp0(stokes_ker); // potential from elem_lst_nbr to X0
  LayerPotenOp0.AddElemList(elem_lst_nbr);
  LayerPotenOp0.SetTargetCoord(X0);
  LayerPotenOp0.SetAccuracy(tol);

  sctl::BoundaryIntegralOp<Real, KerType> LayerPotenOp_proxy(stokes_ker); // potential from elem_lst0 to proxy points
  LayerPotenOp_proxy.AddElemList(elem_lst0);
  LayerPotenOp_proxy.SetTargetCoord(X_proxy);
  LayerPotenOp_proxy.SetAccuracy(tol);

  // periodized layer potential operator
  const auto BIO = [&LayerPotenOp0,&LayerPotenOp_proxy,&X0,&Nrepeat](sctl::Vector<Real>* U, const sctl::Vector<Real>& sigma) {
    const sctl::Long N = sigma.Dim();

    sctl::Vector<Real> sigma_nbr(Nrepeat*N); // repeat sigma Nrepeat times
    for (sctl::Long k = 0; k < Nrepeat; k++) {
      for (sctl::Long i = 0; i < N; i++) {
        sigma_nbr[k*N+i] = sigma[i];
      }
    }

    U->SetZero();
    LayerPotenOp0.ComputePotential(*U, sigma_nbr);
    if (U->Dim() == N && std::is_same<KerType,Stokes3D_CF>::value) (*U) -= sigma*0.5; // for double-layer

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
  solver(&sigma, BIO, -bg_flow(X0), gmres_tol);
  elem_lst0.WriteVTK("vis/sigma", sigma);

  { // Evaluate in interior, compute error and write visualization
    VolumeVis<Real> cube(elem_lst0, comm);
    X0 = cube.GetCoord(); // set new target coordinates
    LayerPotenOp0.SetTargetCoord(X0);
    sctl::Vector<Real> U;
    BIO(&U, sigma);
    U += bg_flow(X0);

    Real max_err = 0;
    const auto err = U - u_ref(X0);
    for (const auto e : err) max_err = std::max<Real>(max_err, fabs(e));
    std::cout<<"Max error = "<<max_err<<'\n';

    cube.WriteVTK("vis/U", U);
    cube.WriteVTK("vis/err", err);
  }
}

int main(int argc, char** argv) {
  using Real = double;
  test<Real>();

  return 0;
}

