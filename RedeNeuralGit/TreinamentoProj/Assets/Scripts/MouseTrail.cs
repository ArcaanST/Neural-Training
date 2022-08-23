using UnityEngine;

[RequireComponent(typeof(TrailRenderer))]
public class MouseTrail : MonoBehaviour
{
    [SerializeField]
    float m_DistanceFromCamera = 10.0f;

    Camera m_Camera;
    Vector3 m_Position;
    TrailRenderer m_trailRenderer;
    // Start is called before the first frame update
    void Start()
    {
        m_Camera = Camera.main;
        m_Position = Vector3.zero;

        m_trailRenderer = GetComponent<TrailRenderer>();
        m_trailRenderer.emitting = false;
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            m_trailRenderer.emitting = true;
        }
        if (Input.GetMouseButtonUp(0))
        {
            m_trailRenderer.emitting = false;
        }

        m_Position = Input.mousePosition;
        m_Position.z = m_DistanceFromCamera;
        m_Position = m_Camera.ScreenToWorldPoint(m_Position);
        transform.position = m_Position;
    }
}
