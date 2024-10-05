#include <mitsuba/core/string.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class RoughConductorHeitz final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture, MicrofacetDistribution)

    RoughConductorHeitz(const Properties &props) : Base(props) {
        std::string material = props.string("material", "none");
        if (props.has_property("eta") || material == "none") {
            m_eta = props.texture<Texture>("eta", 0.f);
            m_k   = props.texture<Texture>("k",   1.f);
            if (material != "none")
                Throw("Should specify either (eta, k) or material, not both.");
        } else {
            std::tie(m_eta, m_k) = complex_ior_from_file<Spectrum, Texture>(props.string("material", "Cu"));
        }

        if (props.has_property("distribution")) {
            std::string distr = string::to_lower(props.string("distribution"));
            if (distr == "beckmann")
                m_type = MicrofacetType::Beckmann;
            else if (distr == "ggx")
                m_type = MicrofacetType::GGX;
            else
                Throw("Specified an invalid distribution \"%s\", must be "
                      "\"beckmann\" or \"ggx\"!", distr.c_str());
        } else {
            m_type = MicrofacetType::Beckmann;
        }

        m_sample_visible = props.get<bool>("sample_visible", true);

        if (props.has_property("alpha_u") || props.has_property("alpha_v")) {
            if (!props.has_property("alpha_u") || !props.has_property("alpha_v"))
                Throw("Microfacet model: both 'alpha_u' and 'alpha_v' must be specified.");
            if (props.has_property("alpha"))
                Throw("Microfacet model: please specify"
                      "either 'alpha' or 'alpha_u'/'alpha_v'.");
            m_alpha_u = props.texture<Texture>("alpha_u");
            m_alpha_v = props.texture<Texture>("alpha_v");
        } else {
            m_alpha_u = m_alpha_v = props.texture<Texture>("alpha", 0.1f);
        }

        if (props.has_property("specular_reflectance"))
            m_specular_reflectance = props.texture<Texture>("specular_reflectance", 1.f);

        m_maxBounces = props.get<int>("maxBounces", 10);

        m_flags = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide;
        if (m_alpha_u != m_alpha_v)
            m_flags = m_flags | BSDFFlags::Anisotropic;

        m_components.clear();
        m_components.push_back(m_flags);
    }

    void traverse(TraversalCallback *callback) override {
        if (m_specular_reflectance)
            callback->put_object("specular_reflectance", m_specular_reflectance.get(), +ParamFlags::Differentiable);
        if (!has_flag(m_flags, BSDFFlags::Anisotropic))
            callback->put_object("alpha",                m_alpha_u.get(),              ParamFlags::Differentiable | ParamFlags::Discontinuous);
        else {
            callback->put_object("alpha_u",              m_alpha_u.get(),              ParamFlags::Differentiable | ParamFlags::Discontinuous);
            callback->put_object("alpha_v",              m_alpha_v.get(),              ParamFlags::Differentiable | ParamFlags::Discontinuous);
        }
        callback->put_object("eta", m_eta.get(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
        callback->put_object("k",   m_k.get(),   ParamFlags::Differentiable | ParamFlags::Discontinuous);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float /* sample1 */,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        BSDFSample3f bs = dr::zeros<BSDFSample3f>();
        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        active &= cos_theta_i > 0.f;

        if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) || dr::none_or<false>(active)))
            return { bs, 0.f };

        /* Construct a microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(m_type,
                                     m_alpha_u->eval_1(si, active),
                                     m_alpha_v->eval_1(si, active),
                                     m_sample_visible);

        bs.eta = 1.f;
        bs.sampled_component = 0;
        bs.sampled_type = +BSDFFlags::GlossyReflection;

        // Inicialização.
        Float hr = INFINITY;           // altura inicial
        Spectrum er(1.f);              // energia inicial
        Vector3f wr = -si.wi;          // direção inicial
        int r = 0;                     // índice para contar o número de reflexões entre as microfacetas

        // Passeio aleatório.
        while (r < m_maxBounces) {
            // Próxima altura.
            hr = sample_height(wr, hr, distr);

            // Verifica se o raio deixa a microssuperfície.
            if (dr::all(isinf(hr))) break;

            // Próxima direção.
            Spectrum weight;
            wr = sample_phase_function(-wr, distr, sample2, m_eta->eval(si, active), m_k->eval(si, active), weight);

            // Próxima energia.
            er = er * weight;

            // Verifica inconsistência númerica. Adaptado do código fornecido por Heitz.
            if (dr::all(dr::isnan(hr)) || dr::all(dr::isnan(wr.x()))) {
                er = Spectrum(0.f);
                wr = Vector3f(0.f, 0.f, 1.f);
                break;
            }

            r++;
        }

        // Direção amostrada.
        bs.wo = wr;
        bs.pdf = pdf(ctx, si, bs.wo, active);

        // Ensure that this is a valid sample
        active &= (bs.pdf != 0.f) && Frame3f::cos_theta(bs.wo) > 0.f;

        return { bs, er & active };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) || dr::none_or<false>(active)))
            return 0.f;

        if (dr::all(wo.z() <= 0))
            return 0.f;

        /* Construct a microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(m_type,
                                     m_alpha_u->eval_1(si, active),
                                     m_alpha_v->eval_1(si, active),
                                     m_sample_visible);

        Spectrum eta = m_eta->eval(si, active);
        Spectrum k = m_k->eval(si, active);

        // Inicialização.
        Float hr = INFINITY;   // altura inicial
        Spectrum er(1.f);      // energia inicial
        Vector3f wr = -si.wi;  // direção inicial
        Spectrum result(0.f);  // valor inicial da BRDF
        int r = 0;             // índice para contar o número de reflexões entre as microfacetas

        // Passeio aleatório.
        while (r < m_maxBounces) {
            // Próxima altura.
            hr = sample_height(wr, hr, distr);

            // Verifica se o raio deixa a microssuperfície.
            if (dr::all(isinf(hr))) break;

            // Cálculo da contribuição da r-ésima reflexão para a energia dispersa.
            Spectrum phase = eval_phase_function(-wr, wo, distr, eta, k);
            Float g1 = G1(wo, hr, distr);
            Spectrum Er = er * phase * g1;

            // Caso a contribuição seja válida, ela é adicionada ao resultado. Adaptado do código fornecido por Heitz.
            result = dr::select(dr::isfinite(Er), result + Er, result);

            // Próxima direção.
            Spectrum weight;
            wr = sample_phase_function(-wr, distr, eta, k, weight);

            // Próxima energia.
            er = er * weight;

            // Verifica inconsistência númerica. Adaptado do código fornecido por Heitz.
            if (dr::all(dr::isnan(hr)) || dr::all(dr::isnan(wr.x())))
                return 0.f;

            r++;
        }

        return result & active;
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        // Calculate the half-direction vector
        Vector3f wh = dr::normalize(wo + si.wi);

        /* Filter cases where the micro/macro-surface don't agree on the side.
           This logic is evaluated in smith_g1() called as part of the eval()
           and sample() methods and needs to be replicated in the probability
           density computation as well. */
        active &= cos_theta_i > 0.f && cos_theta_o > 0.f &&
                  dr::dot(si.wi, wh) > 0.f && dr::dot(wo, wh) > 0.f;

        if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) || dr::none_or<false>(active)))
            return 0.f;

        /* Construct a microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(m_type,
                                     m_alpha_u->eval_1(si, active),
                                     m_alpha_v->eval_1(si, active),
                                     m_sample_visible);

        // Soma da PDF da BRDF de dispersão simples (retirada do método pdf da BRDF
        // 'roughconductor' do Mitsuba) com um fator para contabilizar a dispersão múltipla.
        Float result = distr.eval(wh) * distr.smith_g1(si.wi, wh) / (4.f * cos_theta_i) + cos_theta_o;

        return dr::select(active, result, 0.f);
    }

    std::pair<Spectrum, Float> eval_pdf(const BSDFContext &ctx,
                                        const SurfaceInteraction3f &si,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        // Calculate the half-direction vector
        Vector3f wh = dr::normalize(wo + si.wi);

        /* Filter cases where the micro/macro-surface don't agree on the side.
           This logic is evaluated in smith_g1() called as part of the eval()
           and sample() methods and needs to be replicated in the probability
           density computation as well. */
        active &= cos_theta_i > 0.f && cos_theta_o > 0.f &&
                  dr::dot(si.wi, wh) > 0.f && dr::dot(wo, wh) > 0.f;

        if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) || dr::none_or<false>(active)))
            return { 0.f, 0.f };

        Spectrum result = eval(ctx, si, wo, active);
        Float pdf_ = pdf(ctx, si, wo, active);

        return { result, pdf_ };
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "RoughConductorHeitz[" << std::endl
            << "  distribution = " << m_type << "," << std::endl
            << "  sample_visible = " << m_sample_visible << "," << std::endl
            << "  alpha_u = " << string::indent(m_alpha_u) << "," << std::endl
            << "  alpha_v = " << string::indent(m_alpha_v) << "," << std::endl;
        if (m_specular_reflectance)
           oss << "  specular_reflectance = " << string::indent(m_specular_reflectance) << "," << std::endl;
        oss << "  eta = " << string::indent(m_eta) << "," << std::endl
            << "  k = " << string::indent(m_k) << std::endl
            << "]";
        return oss.str();
    }

private:
    // Implementação da equação 2.20.
    Float G1(const Vector3f &wi, const Float h, const MicrofacetDistribution &distr) const {
        // Tratamento de casos especiais, foi adaptado do código fornecido por Heitz.
        if (dr::all(wi.z() > 0.9999f)) return 1.f;
        if (dr::all(wi.z() <= 0.f)) return 0.f;

        return dr::pow(C1(h), distr.lambda(wi));
    }

    // Implementação do algoritmo 1.
    Float sample_height(const Vector3f &wr, const Float hr, const MicrofacetDistribution &distr) const {
        Float U = generateRandomNumber();

        // Tratamento de casos especiais, foi adaptado do código fornecido por Heitz.
        if (dr::all(wr.z() > 0.9999f))
            return INFINITY;
        if (dr::all(wr.z() < -0.9999f))
            return invC1(U * C1(hr));
        if (dr::all(dr::abs(wr.z()) < 0.0001f))
            return hr;

        // Probabilidade da interseção entre o raio e a microssuperfície.
        const Float g1 = G1(wr, hr, distr);

        // Se o raio sair da microssuperfície, a altura amostrada é infinita.
        if (dr::all(U > 1.f - g1)) return INFINITY;

        // Altura amostrada.
        return invC1(C1(hr) / dr::pow(1.f - U, 1.f / distr.lambda(wr)));
    }

    // Calcula o valor da equação 3.15.
    Spectrum eval_phase_function(const Vector3f &wi, const Vector3f &wo, const MicrofacetDistribution &distr, const Spectrum &eta, const Spectrum &k) const {
        // Tratamento de caso especial, foi adaptado do código fornecido por Heitz.
        if (dr::all(wi.z() > 0.9999f)) return 0.f;

        // Determinação do meio-vetor.
        const Vector3f wh = dr::normalize(wi + wo);

        if (dr::all(wh.z() < 0.f)) return 0.f;

        // Calcula o valor da equação 3.12.
        Float D_wi = dr::maximum(0.f, dr::dot(wi, wh)) * distr.eval(wh) / (wi.z() * (1 + distr.lambda(wi)));

        dr::Complex<UnpolarizedSpectrum> eta_c(eta, k);
        Spectrum F = fresnel_conductor(UnpolarizedSpectrum(dr::dot(wi, wh)), eta_c);

        return F * D_wi / (4.f * dr::dot(wi, wh));
    }

    // Gera um número aleatório. Foi retirado do código fornecido por Heitz.
    float generateRandomNumber() const {
        const float U = ((float) rand()) / (float) RAND_MAX;
        return U;
    }

    // CDF de uma distribuição uniforme de alturas. Foi retirado do código fornecido por Heitz.
    Float C1(const Float h) const {
        return dr::minimum(1.f, dr::maximum(0.f, 0.5f * (h + 1.f)));
    }

    // A inversa da CDF de uma distribuição uniforme de alturas. Foi retirado do código fornecido por Heitz.
    Float invC1(const Float U) const {
        return dr::maximum(-1.0f, dr::minimum(1.0f, 2.0f * U - 1.0f));
    }

    // Implementação do algoritmo 3.
    // Realiza a amostragem por importância da função de fase para o condutor e devolve uma
    // direção de saída.
    Vector3f sample_phase_function(const Vector3f &wi, const MicrofacetDistribution &distr, const Point2f &sample2, const Spectrum &eta, const Spectrum &k, Spectrum &weight) const {
        // Amostra de uma normal para a microfaceta.
        Vector3f wm;
        std::tie(wm, std::ignore) = distr.sample(wi, sample2);

        // Reflexão de wi em relação à normal wm para obter a direção de saída, wo.
        const Vector3f wo = -wi + 2.0f * wm * dr::dot(wi, wm);

        // Peso da amostra.
        dr::Complex<UnpolarizedSpectrum> eta_c(eta, k);
        weight = fresnel_conductor(UnpolarizedSpectrum(dr::dot(wi, wm)), eta_c);

        return wo;
    }

    // Sobrecarga de método para quando um ponto de amostra não é fornecido.
    Vector3f sample_phase_function(const Vector3f &wi, const MicrofacetDistribution &distr, const Spectrum &eta, const Spectrum &k, Spectrum &weight) const {
        Point2f sample2(generateRandomNumber(), generateRandomNumber());

        // Amostra de uma normal para a microfaceta.
        Vector3f wm;
        std::tie(wm, std::ignore) = distr.sample(wi, sample2);

        // Reflexão de wi em relação à normal wm para obter a direção de saída, wo.
        const Vector3f wo = -wi + 2.0f * wm * dr::dot(wi, wm);

        // Peso da amostra.
        dr::Complex<UnpolarizedSpectrum> eta_c(eta, k);
        weight = fresnel_conductor(UnpolarizedSpectrum(dr::dot(wi, wm)), eta_c);

        return wo;
    }

    MI_DECLARE_CLASS()

private:
    /// Specifies the type of microfacet distribution
    MicrofacetType m_type;
    /// Anisotropic roughness values
    ref<Texture> m_alpha_u, m_alpha_v;
    /// Importance sample the distribution of visible normals?
    bool m_sample_visible;
    /// Relative refractive index (real component)
    ref<Texture> m_eta;
    /// Relative refractive index (imaginary component).
    ref<Texture> m_k;
    /// Specular reflectance component
    ref<Texture> m_specular_reflectance;
    /// Scattering Order
    int m_maxBounces;
};

MI_IMPLEMENT_CLASS_VARIANT(RoughConductorHeitz, BSDF)
MI_EXPORT_PLUGIN(RoughConductorHeitz, "Rough conductor Heitz")
NAMESPACE_END(mitsuba)
