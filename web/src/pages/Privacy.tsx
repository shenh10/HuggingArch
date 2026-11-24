import { useEffect } from "react";
import { Link } from "react-router-dom";

/** Static policy at /privacy. Text matches vanilla frontend/privacy.html. */
export function Privacy() {
  useEffect(() => {
    document.title = "Privacy · HuggingArch";
    return () => {
      document.title = "HuggingArch";
    };
  }, []);
  return (
    <div className="bg-gray-50 min-h-screen">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link to="/" className="inline-flex items-center gap-2 text-lg font-semibold text-gray-900 hover:text-amber-600">
            <img src="/static/logo.png" alt="HuggingArch" className="w-7 h-7" />
            HuggingArch
          </Link>
          <Link to="/" className="text-sm text-gray-600 hover:text-amber-600">Back to app</Link>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-10 break-words">
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">Privacy Policy</h1>
        <p className="text-xs text-gray-500 mb-8">Last updated 2026-05-05</p>

        <p className="text-sm text-gray-700 leading-relaxed">
          HuggingArch is an open-source architecture analyzer and inference planner
          for HuggingFace transformer models. This page describes what data the
          service collects, why, and how to remove it.
        </p>

        <h2 className="text-base font-semibold text-gray-900 mt-8 mb-2">What we collect</h2>

        <p className="text-sm text-gray-700 leading-relaxed mb-2">When you use the site without signing in:</p>
        <ul className="text-sm text-gray-700 list-disc list-inside space-y-1 mb-4">
          <li>The model id you analyze, the endpoint you hit (<code>/analyze</code>, <code>/inference/estimate</code>), and how long the request took. No user identifier is attached. Used for site-wide popularity and latency analytics.</li>
          <li>Your IP address and User-Agent header, only as transient input to the rate limiter. Not stored.</li>
        </ul>

        <p className="text-sm text-gray-700 leading-relaxed mb-2">When you sign in (email/password, GitHub, or Google):</p>
        <ul className="text-sm text-gray-700 list-disc list-inside space-y-1 mb-4">
          <li><strong>Profile</strong>: username, full name, bio, avatar URL — all editable from the profile page. Synced from your OAuth provider on first sign-in.</li>
          <li><strong>Login history</strong>: IP address and User-Agent on each session creation. Used for active-user analytics; visible only to administrators of this deployment.</li>
          <li><strong>Per-call activity</strong>: the same analyze events as above, but tagged with your account id. Used for per-user usage and admin analytics.</li>
        </ul>

        <h2 className="text-base font-semibold text-gray-900 mt-8 mb-2">Where it lives</h2>
        <p className="text-sm text-gray-700 leading-relaxed">
          All user data lives in a Supabase Postgres database controlled by the
          operator of this deployment. Authentication is handled by Supabase
          Auth. We do not sell, share, or transfer your data to third parties.
        </p>

        <h2 className="text-base font-semibold text-gray-900 mt-8 mb-2">Third-party providers</h2>
        <ul className="text-sm text-gray-700 list-disc list-inside space-y-1 mb-4">
          <li><strong>Supabase</strong> — auth + database hosting. <a className="text-amber-700 hover:underline" href="https://supabase.com/privacy" target="_blank" rel="noopener">supabase.com/privacy</a></li>
          <li><strong>Cloudflare</strong> — network edge: every request passes through it (it sees your IP address). <a className="text-amber-700 hover:underline" href="https://www.cloudflare.com/privacypolicy/" target="_blank" rel="noopener">cloudflare.com/privacypolicy</a></li>
          <li><strong>HuggingFace</strong> — public model metadata fetched on every analyze request (no PII sent). <a className="text-amber-700 hover:underline" href="https://huggingface.co/privacy" target="_blank" rel="noopener">huggingface.co/privacy</a></li>
          <li><strong>GitHub / Google</strong> — only when you choose to sign in via OAuth. Each provider has its own privacy policy.</li>
        </ul>

        <h2 className="text-base font-semibold text-gray-900 mt-8 mb-2">How to delete your data</h2>
        <p className="text-sm text-gray-700 leading-relaxed">
          Email the operator of this deployment to request deletion of your
          profile, login history, and account. Past anonymous usage events are
          retained but cannot be linked back to you once your user id is removed
          (analyze logs cascade to <code>user_id = NULL</code>).
        </p>

        <h2 className="text-base font-semibold text-gray-900 mt-8 mb-2">Cookies</h2>
        <p className="text-sm text-gray-700 leading-relaxed">
          The site uses one cookie set by Supabase Auth to keep you signed in
          across page loads. No analytics or advertising cookies are used.
        </p>

        <h2 className="text-base font-semibold text-gray-900 mt-8 mb-2">Changes to this policy</h2>
        <p className="text-sm text-gray-700 leading-relaxed">
          Material changes will be reflected by an update to the date at the top
          of this page. The current version is always available at
          {" "}<code>https://www.huggingarch.com/privacy</code>.
        </p>

        <h2 className="text-base font-semibold text-gray-900 mt-8 mb-2">Source code</h2>
        <p className="text-sm text-gray-700 leading-relaxed">
          HuggingArch is open source under the MIT license. The exact
          data flow described above can be audited at
          {" "}
          <a className="text-amber-700 hover:underline" href="https://github.com/shenh10/HuggingArch" target="_blank" rel="noopener">github.com/shenh10/HuggingArch</a>.
        </p>
      </main>
    </div>
  );
}
