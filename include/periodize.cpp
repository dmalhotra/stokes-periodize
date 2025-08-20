#include <tuple>

template <class Real> class PeriodizeOp {
  static constexpr Real tol = sctl::machine_eps<Real>()*64; // tolerance for pseudo-inverse
  static constexpr sctl::Long m0 = 20; // multipole order
  static constexpr sctl::Integer COORD_DIM = 3;

  using KerM2M = sctl::Stokes3D_FxU;
  using KerM2L = sctl::Stokes3D_FxU;
  using KerL2L = sctl::Stokes3D_FxU;

  public:

    static constexpr sctl::Long Nsurf() {
      return 6*(m0-1)*(m0-1)+2;
    }
    static sctl::Vector<Real> uc_surf(const Real box_length=1, const sctl::Vector<Real>& X0 = sctl::Vector<Real>{0,0,0}) {
      return proxy_surf(box_length*2.95, X0);
    }
    static sctl::Vector<Real> ue_surf(const Real box_length=1, const sctl::Vector<Real>& X0 = sctl::Vector<Real>{0,0,0}) {
      return proxy_surf(box_length*1.05, X0);
    }
    static sctl::Vector<Real> dc_surf(const Real box_length=1, const sctl::Vector<Real>& X0 = sctl::Vector<Real>{0,0,0}) {
      return proxy_surf(box_length*1.05, X0);
    }
    static sctl::Vector<Real> de_surf(const Real box_length=1, const sctl::Vector<Real>& X0 = sctl::Vector<Real>{0,0,0}) {
      return proxy_surf(box_length*2.95, X0);
    }

    static std::tuple<sctl::Matrix<Real>, sctl::Matrix<Real>> UC2UE(const Real box_length=1) {
      sctl::Profile::Scoped prof(__FUNCTION__);
      const auto Xc = uc_surf(box_length);
      const auto Xe = ue_surf(box_length);
      const KerM2M ker_m2m;

      sctl::Matrix<Real> Me2c, U,S,Vt, Mc2e0, Mc2e1;
      ker_m2m.KernelMatrix<Real,true>(Me2c, Xc, Xe, sctl::Vector<Real>());
      sctl::Matrix<Real>(Me2c).SVD(U,S,Vt);

      Real max_val = 0;
      for (sctl::Long i = 0; i < std::min(S.Dim(0),S.Dim(1)); i++) max_val = std::max<Real>(max_val, sctl::fabs(S[i][i]));
      for (sctl::Long i = 0; i < std::min(S.Dim(0),S.Dim(1)); i++) S[i][i] = (sctl::fabs(S[i][i]) < max_val*tol ? 0 : 1/S[i][i]);

      Mc2e0 = Vt.Transpose();
      Mc2e1 = S * U.Transpose();
      return std::make_tuple(Mc2e0, Mc2e1);
    }

    static std::tuple<sctl::Matrix<Real>, sctl::Matrix<Real>> DC2DE(const Real box_length=1) {
      sctl::Profile::Scoped prof(__FUNCTION__);
      const auto Xc = dc_surf(box_length);
      const auto Xe = de_surf(box_length);
      const KerL2L ker_l2l;

      sctl::Matrix<Real> Me2c, U,S,Vt, Mc2e0, Mc2e1;
      ker_l2l.KernelMatrix<Real,true>(Me2c, Xc, Xe, sctl::Vector<Real>());
      sctl::Matrix<Real>(Me2c).SVD(U,S,Vt);

      Real max_val = 0;
      for (sctl::Long i = 0; i < std::min(S.Dim(0),S.Dim(1)); i++) max_val = std::max<Real>(max_val, sctl::fabs(S[i][i]));
      for (sctl::Long i = 0; i < std::min(S.Dim(0),S.Dim(1)); i++) S[i][i] = (sctl::fabs(S[i][i]) < max_val*tol ? 0 : 1/S[i][i]);

      Mc2e0 = Vt.Transpose() * S;
      Mc2e1 = U.Transpose();
      return std::make_tuple(Mc2e0, Mc2e1);
    }

    static sctl::Matrix<Real> BC_UE2DC() {
      static const auto M = BC_UE2DC_helper();
      return M;
    }

  private:

    static sctl::Vector<Real> proxy_surf(const Real box_length, const sctl::Vector<Real>& Xc) {
      static const sctl::Vector<Real> X0 = []() {
        sctl::Vector<Real> X;
        for (sctl::Long i0 = 0; i0 < m0; i0++) {
          for (sctl::Long i1 = 0; i1 < m0; i1++) {
            for (sctl::Long i2 = 0; i2 < m0; i2++) {
              if (i0==0 || i0==m0-1 || i1==0 || i1==m0-1 || i2==0 || i2==m0-1) {
                const Real x = i0/(Real)(m0-1);
                const Real y = i1/(Real)(m0-1);
                const Real z = i2/(Real)(m0-1);
                X.PushBack(x-0.5);
                X.PushBack(y-0.5);
                X.PushBack(z-0.5);
              }
            }
          }
        }
        return X;
      }();
      SCTL_ASSERT(X0.Dim() == Nsurf()*COORD_DIM);

      sctl::Vector<Real> X(Nsurf()*COORD_DIM);
      for (sctl::Long i = 0; i < Nsurf(); i++) {
        for (sctl::Long k = 0; k < COORD_DIM; k++) {
          X[i*COORD_DIM+k] = X0[i*COORD_DIM+k]*box_length + Xc[k];
        }
      }
      return X;
    }

    static sctl::Matrix<Real> BC_UE2DC_helper() {
      sctl::Profile::Scoped prof(__FUNCTION__);

      sctl::Matrix<Real> M;
      M.template Read<sctl::QuadReal>("data/Mbc_ue2dc_1d.mat");
      if (M.Dim(0) || M.Dim(1)) return M;

      const KerM2L ker_m2l;
      const sctl::Integer kdim[2] = {KerM2L::SrcDim(), KerM2L::TrgDim()};
      const auto X0 = dc_surf();

      sctl::Matrix<Real> M2M(Nsurf()*kdim[0], Nsurf()*kdim[0]);
      M.ReInit(Nsurf()*kdim[0], Nsurf()*kdim[1]);
      M2M.SetZero();
      M.SetZero();

      for (sctl::Long i = 0; i < M2M.Dim(0); i++) { // M2M <-- Identity
        M2M[i][i] = 1;
      }

      for (sctl::Long l = 0; l < 30; l++) { // tree-code (hierarchical) summation
        std::cout<<"level = "<<l<<'\n';
        const sctl::Long box_length = ((sctl::Long)1) << l;

        sctl::Matrix<Real> M0, M1;
        ker_m2l.KernelMatrix<Real,true>(M0, X0, ue_surf(box_length, sctl::Vector<Real>{ (Real)1.5*box_length+0.5,0,0}), sctl::Vector<Real>());
        ker_m2l.KernelMatrix<Real,true>(M1, X0, ue_surf(box_length, sctl::Vector<Real>{-(Real)1.5*box_length-0.5,0,0}), sctl::Vector<Real>());
        M += M2M * (M0 + M1);

        ker_m2l.KernelMatrix<Real,true>(M0, uc_surf(2*box_length), ue_surf(box_length, sctl::Vector<Real>{ (Real)0.5*box_length,0,0}), sctl::Vector<Real>());
        ker_m2l.KernelMatrix<Real,true>(M1, uc_surf(2*box_length), ue_surf(box_length, sctl::Vector<Real>{-(Real)0.5*box_length,0,0}), sctl::Vector<Real>());
        const auto [M_uc2ue0, M_uc2ue1] = PeriodizeOp<Real>::UC2UE(2*box_length);
        M2M = M2M * (((M0+M1) * M_uc2ue0) * M_uc2ue1);
      }
      //for (sctl::Long i = -1000; i <= 1000; i++) { // direct summation
      //  if (abs(i) >= 2) {
      //    sctl::Matrix<Real> M_;
      //    ker_m2l.KernelMatrix<Real,true>(M_, X0, ue_surf(1, sctl::Vector<Real>{(Real)i,0,0}), sctl::Vector<Real>());
      //    M += M_;
      //  }
      //}

      M.template Write<sctl::QuadReal>("data/Mbc_ue2dc_1d.mat");
      return M;
    }

};

template <class Real> const sctl::Vector<Real>& Periodize<Real>::GetProxySurf() {
  static const auto X = PeriodizeOp<Real>::uc_surf(1, sctl::Vector<Real>{0.5,0.5,0.5});
  return X;
}

template <class Real> void Periodize<Real>::EvalFarField(sctl::Vector<Real>& U_far, const sctl::Vector<Real>& Xt, const sctl::Vector<Real>& U_proxy) {
  const auto& Mbc0 = GetMat_UC2DE0();
  const auto& Mbc1 = GetMat_UC2DE1();
  const sctl::Long N = Mbc0.Dim(0);
  SCTL_ASSERT(U_proxy.Dim() == N);

  // Compute the equivalent density at proxy points
  auto proxy_density = (sctl::Matrix<Real>(1,N,(sctl::Iterator<Real>)U_proxy.begin(),false) * Mbc0) * Mbc1;

  // Evaluate the potential from proxy points at the targets Xt
  U_far = 0;
  static const sctl::Stokes3D_FxU stokeslet;
  stokeslet.template Eval<Real,true>(U_far, Xt, GetProxySurf(), sctl::Vector<Real>(), sctl::Vector<Real>(N,proxy_density.begin(),false));
}

template <class Real> const sctl::Matrix<Real>& Periodize<Real>::GetMat_UC2DE0() {
  static sctl::Matrix<Real> Mbc = [](){
    sctl::Matrix<Real> M;
    M.template Read<sctl::QuadReal>("data/Mbc_uc2de0_1d.mat");
    if (M.Dim(0) || M.Dim(1)) return M;

    const auto [M_uc2ue0, M_uc2ue1] = PeriodizeOp<Real>::UC2UE();
    const auto [M_dc2de0, M_dc2de1] = PeriodizeOp<Real>::DC2DE();
    const auto Mbc_ue2dc = PeriodizeOp<Real>::BC_UE2DC();
    M = (M_uc2ue0 * (M_uc2ue1 * Mbc_ue2dc)) * M_dc2de0;
    M.template Write<sctl::QuadReal>("data/Mbc_uc2de0_1d.mat");
    return M;
  }();
  return Mbc;
}

template <class Real> const sctl::Matrix<Real>& Periodize<Real>::GetMat_UC2DE1() {
  static sctl::Matrix<Real> Mbc = [](){
    sctl::Matrix<Real> M;
    M.template Read<sctl::QuadReal>("data/Mbc_uc2de1_1d.mat");
    if (M.Dim(0) || M.Dim(1)) return M;

    const auto [M_dc2de0, M_dc2de1] = PeriodizeOp<Real>::DC2DE();
    M_dc2de1.template Write<sctl::QuadReal>("data/Mbc_uc2de1_1d.mat");
    return M_dc2de1;
  }();
  return Mbc;
}

